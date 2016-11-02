package ptemplin.nlp.asr.io;

import javax.sound.sampled.AudioFormat;

public class SpeechSample {

	public byte[] data;
	public AudioFormat format;
	public SpeechSample(byte[] data, AudioFormat format) {
		this.data = data;
		this.format = format;
	}
	
	public int[] getDataAsInts() {
		int[] speechData = new int[data.length/2];
		for (int i = 0; i < speechData.length; i++) {
			int littleVal = data[i*2];
			if (littleVal < 0) {
				littleVal = 256 + littleVal;
			}
			int bigVal = data[i*2+1];
			speechData[i] = bigVal * 256 + littleVal;
		}
		return speechData;
	}
	
}
