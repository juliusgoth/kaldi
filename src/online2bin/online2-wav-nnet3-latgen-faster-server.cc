// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
//#include "online2bin/StdCapture.hpp"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "thread/kaldi-thread.h"
#include <cstdio>

// boost IO stuff
#include <boost/iostreams/concepts.hpp> 
#include <boost/iostreams/stream_buffer.hpp>
#include <iostream>

#include "server_http.hpp"
#include "client_http.hpp"


#include "fst/script/print-impl.h"


namespace kaldi {

typedef SimpleWeb::Server<SimpleWeb::HTTP> HttpServer;

namespace bio = boost::iostreams;

class MySink : public bio::sink
{
public:
    std::string buf;
    std::streamsize write(const char* s, std::streamsize n)
    {
      std::string sb(s, n);
      std::cout << "writing to stream" << sb << std::endl;
      //std::cout << buf << std::endl;
        //Do whatever you want with s
        //...
        return n;
    }
};

void GetDiagnosticsAndPrintOutput(std::ostream &os,
                                  const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  std::locale loc;
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";

  if (word_syms != NULL) {
    //std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      os << s << ' ';
    }
    os << std::endl;
  }
}

}

    std::string nnet3_rxfilename,
        fst_rxfilename,
        spk2utt_rspecifier,
        wav_rspecifier,
        clat_wspecifier;

//static SequentialTokenVectorReader spk2utt_reader;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    HttpServer server;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n";

    ParseOptions po(usage);

    static std::string word_syms_rxfilename;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    static OnlineNnet2FeaturePipelineConfig feature_opts;
    static nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    static LatticeFasterDecoderConfig decoder_opts;
    static OnlineEndpointConfig endpoint_opts;

    static BaseFloat chunk_length_secs = 0.18;
    static bool do_endpointing = false;
    static bool online = true;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    po.Register("online", &online,
                "You can set this to false to disable online iVector estimation "
                "and have all the data for each utterance used, even at "
                "utterance start.  This is useful where you just want the best "
                "results and don't care about online operation.  Setting this to "
                "false has the same effect as setting "
                "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                "in the file given to --ivector-extraction-config, and "
                "--chunk-length=-1.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    endpoint_opts.Register(&po);


    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      return 1;
    }

    /*std::string */nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        spk2utt_rspecifier = po.GetArg(3),
        wav_rspecifier = po.GetArg(4),
        clat_wspecifier = po.GetArg(5);


    static OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

    if (!online) {
      feature_info.ivector_extractor_info.use_most_recent_ivector = true;
      feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
      chunk_length_secs = -1.0;
    }

    static TransitionModel trans_model;
    static nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    // this object contains precomputed stuff that is used by all decodable
    // objects.  It takes a pointer to am_nnet because if it has iVectors it has
    // to modify the nnet to accept iVectors at intervals.
    static nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                        &am_nnet);


    static fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(fst_rxfilename);

    static fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    //static int32 num_done = 0, num_err = 0;
    static double tot_like = 0.0;
    static int64 num_frames = 0;


    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    //spk2utt_reader.Next();
    std::cout << "key: " << spk2utt_reader.Key() << std::endl;
    const std::vector<std::string> &uttlist = spk2utt_reader.Value();
    static std::string utt = uttlist[0];
    std::cout << "utt: " << utt << std::endl;
    static  OnlineIvectorExtractorAdaptationState adaptation_state(
          feature_info.ivector_extractor_info);


    server.config.port=8181;
    
    //Add resources using path-regex and method-string, and an anonymous function
    //POST-example for the path /string, responds the posted string
    server.resource["^/asr$"]["POST"]=[](shared_ptr<HttpServer::Response> response, shared_ptr<HttpServer::Request> request) {
        //Retrieve string:
        auto content=request->content.string();
        std::string resptext = "Done.";
        std::string outfile = "/home/jugoth/kaldi/egs/aspire/s5/output.wav";
        //request->content.string() is a convenience function for:
        //stringstream ss;
        //ss << request->content.rdbuf();
        //auto content=ss.str();
        
        std::ofstream out(outfile);
        out << content;
        out.flush();
        out.close();
        // write out response
        //std::stringstream outss;
        RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
        CompactLatticeWriter clat_writer(clat_wspecifier);

        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);

        OnlineTimingStats timing_stats;

        cout << "Getting wav data...";

        const WaveData &wave_data = wav_reader.Value(utt);
        cout << "Done." << endl;
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);


        OnlineSilenceWeighting silence_weighting(
            trans_model,
            feature_info.silence_weighting_config);

        OnlineTimer decoding_timer(utt);

        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
          chunk_length = int32(samp_freq * chunk_length_secs);
          if (chunk_length == 0) chunk_length = 1;
        } else {
          chunk_length = std::numeric_limits<int32>::max();
        }

        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;
        SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                            decodable_info,
                                            *decode_fst, &feature_pipeline);

        while (samp_offset < data.Dim()) {
          int32 samp_remaining = data.Dim() - samp_offset;
          int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                         : samp_remaining;

          SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
          feature_pipeline.AcceptWaveform(samp_freq, wave_part);

          samp_offset += num_samp;
          decoding_timer.WaitUntil(samp_offset / samp_freq);

/*
*/
          if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
          }
/*          
*/
          if (silence_weighting.Active() &&
              feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
          }

          decoder.AdvanceDecoding();

          if (do_endpointing && decoder.EndpointDetected(endpoint_opts))
            break;
        }
        decoder.FinalizeDecoding();

        CompactLattice clat;
        bool end_of_utterance = true;
        decoder.GetLattice(end_of_utterance, &clat);

        //streambuf* oldCoutStreamBuf = std::cerr.rdbuf();
        ostringstream strCout;
        //std::cerr.rdbuf( strCout.rdbuf() );

        GetDiagnosticsAndPrintOutput(strCout, utt, word_syms, clat,
                                     &num_frames, &tot_like);

        //std::cerr.rdbuf(oldCoutStreamBuf);

        //CompactLatticeHolder::Write(outss, true, clat);
        //decoding_timer.OutputStats(&timing_stats);

        // In an application you might avoid updating the adaptation state if
        // you felt the utterance had low confidence.  See lat/confidence.h
        feature_pipeline.GetAdaptationState(&adaptation_state);

        // we want to output the lattice with un-scaled acoustics.
/*
        BaseFloat inv_acoustic_scale =
            1.0 / decodable_opts.acoustic_scale;
        ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);
*/
        //MySink sink;
        //bio::stream_buffer<MySink> sb;
        //sb.open(sink);
        //std::streambuf * oldbuf = std::cerr.rdbuf(&sb);
        //std::cerr.rdbuf(oldbuf);
        /**/
        /**/
        //StdCapture capture;
        //std::stringstream* outss = std::cout.rdbuf
        //cout.rdbuf(outss.rdbuf());
        //capture.BeginCapture();

        /*outss << *///clat_writer.Write(utt, clat);
        //capture.EndCapture();

        //std::stringstream outss = clat_writer.getStream();

        //outstring = outss.str();
        //outstring = capture.GetCapture();
        
        /*
        std::string outstring("Done.");        
        */

        //std::string outstring = strCout.str();

        /*
        bool acceptor = false, write_one = false;
        fst::FstPrinter<LatticeArc> printer(clat, clat.InputSymbols(),
                                            clat.OutputSymbols(),
                                            NULL, acceptor, write_one, "\t");
        printer.Print(&strCout, "<unknown>");
        */        
        //WriteCompactLattice(strCout, false, clat);

        //outstring = "Done.";
        //cout << clat_writer.outstream.str() << endl;
        *response << "HTTP/1.1 200 OK\r\nContent-Length: " << strCout.str().size() << "\r\n\r\n" << strCout.str();
        
        // Alternatively, use one of the convenience functions, for instance:
        // response->write(content);
    };
    server.on_error=[](std::shared_ptr<HttpServer::Request> /*request*/, const SimpleWeb::error_code &/*ec*/) {
        // handle errors here
    };
    thread server_thread([&server](){
        //Start server
        server.start();
    });
    
    //Wait for server to start so that the client can connect
    std::cout << "Preparing to wait" << std::endl;
    this_thread::sleep_for(chrono::seconds(1));
    server_thread.join();
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
