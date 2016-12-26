package ptemplin.nlp.asr.io;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

import static ptemplin.nlp.asr.application.Constants.PHONEME_SPEECH_SAMPLE_DIR;
import static ptemplin.nlp.asr.application.Constants.WORD_SPEECH_SAMPLE_DIR;

public class SpeechFileReader {

	private static final String PHONEME_RECORDING_FILEPATH_SUFFIX = "_recording.wav";
	private static final String SPEECH_DATA_FILE_EXTENSION = ".wav";

	private static final int INPUT_STREAM_BUFFER_SIZE = 128000;
	
	public SpeechSample readSpeechFileForTrainingData(String dir, int dataNum)
			throws UnsupportedAudioFileException, IOException {
		String fileName = WORD_SPEECH_SAMPLE_DIR + dir + "/" + dataNum + SPEECH_DATA_FILE_EXTENSION;
		return readSpeechFile(fileName);
	}

	public SpeechSample readSpeechFileForPhone(String phoneme) throws UnsupportedAudioFileException, IOException {
		String fileName = PHONEME_SPEECH_SAMPLE_DIR + phoneme.toString() + PHONEME_RECORDING_FILEPATH_SUFFIX;
		return readSpeechFile(fileName);
	}
	
	public SpeechSample readSpeechFile(String filePath) throws IOException, UnsupportedAudioFileException {
		File soundFile = new File(filePath);
		AudioInputStream audioStream = AudioSystem.getAudioInputStream(soundFile);
		AudioFormat format = audioStream.getFormat();
		// read the contents of the stream buffer by buffer
		int nBytesRead = 0;
		byte[] abData = new byte[INPUT_STREAM_BUFFER_SIZE];
		ArrayList<Byte> fullSample = new ArrayList<>();
		while (nBytesRead != -1) {
			nBytesRead = audioStream.read(abData, 0, abData.length);
			if (nBytesRead >= 0) {
				for (int i = 0; i < nBytesRead; i++) {
					fullSample.add(abData[i]);
				}
			}
		}
		byte[] fullSampleArr = new byte[fullSample.size()];
		for (int i = 0; i < fullSample.size(); i++) {
			fullSampleArr[i] = fullSample.get(i);
		}
		SpeechSample sample = new SpeechSample(fullSampleArr, format);
		isolateLeftChannel(sample);
		decreaseSampleRate(sample);
		return sample;
	}
	
	private static void isolateLeftChannel(SpeechSample sample) {
		byte[] leftChannel = new byte[sample.data.length/2];
		for (int i = 0; i < sample.data.length/2; i+=2) {
			leftChannel[i] = sample.data[i*2];
			leftChannel[i+1] = sample.data[i*2 + 1];
		}
		sample.data = leftChannel;
	}
	
	private static void decreaseSampleRate(SpeechSample sample) {
		byte[] reducedSampleData = new byte[sample.data.length/4];
		for (int i = 0; i < sample.data.length/4; i+=2) {
			reducedSampleData[i] = sample.data[i*4];
			reducedSampleData[i+1] = sample.data[i*4+1];
		}
		sample.data = reducedSampleData;
	}
	
}
