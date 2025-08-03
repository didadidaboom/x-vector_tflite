import 'dart:math';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:flutter_sound_processing/flutter_sound_processing.dart';

class VoiceMatch {
  Interpreter? _interpreter;

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('x_vector.tflite');
    var inputDetails = _interpreter!.getInputDetails();
    _interpreter!.resizeInputTensor(inputDetails[0]['index'], [1, -1, 24]);
    _interpreter!.allocateTensors();
  }

  Future<List<double>> extractEmbedding(String audioPath) async {
    var audio = FlutterSound();
    await audio.openAudioSession();
    var signal = await audio.startPlayer(audioPath);
    var mfcc = await FlutterSoundProcessing().extractMFCC(
      signal,
      sampleRate: 16000,
      nMfcc: 24,
      hopLength: 160,
      winLength: 400
    );
    print('MFCC 形状: ${mfcc.shape}');
    var inputMfcc = mfcc.reshape([1, mfcc.length, 24]);
    var output = List.filled(512, 0.0).reshape([1, 512]);
    _interpreter!.setTensor(_interpreter!.getInputIndex('serving_default_mfcc:0'), inputMfcc);
    _interpreter!.invoke();
    output = _interpreter!.getOutputTensor(_interpreter!.getOutputIndex('StatefulPartitionedCall:0'));
    await audio.closeAudioSession();
    return output[0];
  }

  double cosineSimilarity(List<double> emb1, List<double> emb2) {
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < emb1.length; i++) {
      dot += emb1[i] * emb2[i];
      norm1 += emb1[i] * emb1[i];
      norm2 += emb2[i] * emb2[i];
    }
    return dot / (sqrt(norm1) * sqrt(norm2));
  }
}