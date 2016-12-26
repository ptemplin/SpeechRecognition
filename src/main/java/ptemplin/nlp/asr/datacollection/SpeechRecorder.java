package ptemplin.nlp.asr.datacollection;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.SourceDataLine;
import javax.sound.sampled.TargetDataLine;

import static ptemplin.nlp.asr.application.Constants.WORD_SPEECH_SAMPLE_DIR;

public class SpeechRecorder {

	private static final int SAMPLE_LENGTH_MILLIS = 2000;
	private static final int SAMPLES_PER_SECOND = 48000;
	private static final int SAMPLE_SIZE_IN_BITS = 16;
	
	private static final int COLLECTION_SIZE = 10;

	private final AudioFormat audioFormat;
	
	public static void main(String[] args) throws InterruptedException {
		// prompt for the directory name
		Scanner scanner = new Scanner(System.in);
		System.out.println("What are you recording?: ");
		String wordDir = scanner.nextLine();
		scanner.close();
		
		SpeechRecorder recorder = new SpeechRecorder();
		for (int i = 1; i <= COLLECTION_SIZE; i++) {
			System.out.println("Sample #" + i);
			byte[] recorded = recorder.record();
			if (recorded != null) {
				recorder.writeAudioToFile(recorded, WORD_SPEECH_SAMPLE_DIR + wordDir + "/", i);
			}
		}
	}
	
	public SpeechRecorder() {
		audioFormat = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED,
				SAMPLES_PER_SECOND, SAMPLE_SIZE_IN_BITS, 2, 4, SAMPLES_PER_SECOND*2, false);
	}
	
	public byte[] record() throws InterruptedException {
		TargetDataLine line;
		DataLine.Info info = new DataLine.Info(TargetDataLine.class, 
		    audioFormat); // format is an AudioFormat object
		if (!AudioSystem.isLineSupported(info)) {
		    // Handle the error ... 
			System.out.println("Unsupported audio format");
			return null;
		}
		// Obtain and open the line.
		try {
		    line = (TargetDataLine) AudioSystem.getLine(info);
		    line.open(audioFormat);
		} catch (LineUnavailableException ex) {
		    // Handle the error ...
			System.out.println("Audio line unavailable");
			return null;
		}
		
		// Assume that the TargetDataLine, line, has already
		// been obtained and opened.
		ByteArrayOutputStream out  = new ByteArrayOutputStream();
		int numBytesRead;
		byte[] data = new byte[line.getBufferSize() / 5];

		// Begin audio capture.
		System.out.println("3...");
		Thread.sleep(1000);
		System.out.println("2...");
		Thread.sleep(1000);
		System.out.println("1...");
		Thread.sleep(1000);
		System.out.println("Recording...");
		line.start();

		long start = System.currentTimeMillis();
		while (System.currentTimeMillis() - start < SAMPLE_LENGTH_MILLIS) {
		   // Read the next chunk of data from the TargetDataLine.
		   numBytesRead =  line.read(data, 0, data.length);
		   // Save this chunk of data.
		   out.write(data, 0, numBytesRead);
		}
		line.close();
		System.out.println("Recording stopped");
		return out.toByteArray();
	}
	
	public void writeAudioToFile(byte[] data, String dir, int recordingNum) {
		File audioDir = new File(dir);
		if (!audioDir.exists()) {
			audioDir.mkdir();
		}
		File audioFile = new File(dir + recordingNum + ".wav");
		ByteArrayInputStream bais = new ByteArrayInputStream(data);
		try (AudioInputStream inputStream = new AudioInputStream(bais, audioFormat, data.length)) {
			AudioSystem.write(inputStream, AudioFileFormat.Type.WAVE, audioFile);
		} catch (IOException ex) {
			ex.printStackTrace();
			System.out.println("Error writing audio data to file at recording num " + recordingNum);
		}
	}
	
	public void playback(byte[] data) {
		try {
			SourceDataLine dataLine = AudioSystem.getSourceDataLine(audioFormat);
			dataLine.open();
			dataLine.start();
			System.out.println("Playing back");
			dataLine.write(data, 0, data.length);
			dataLine.drain();
			dataLine.close();
		} catch (LineUnavailableException ex) {
			ex.printStackTrace();
		}
	}
	
}
