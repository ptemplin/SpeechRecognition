package ptemplin.nlp.asr.frontend;

import java.util.ArrayList;
import java.util.List;

public class SilenceProcessor {
	
	private static final int LOWPASS_N = 50;
	private static final double SILENCE_THRESHOLD_RATIO = 1.0/5;

	public static int[] trimSilence(int[] speechSignal) {
		int[] speechEnvelope = new int[speechSignal.length];
		// low-pass filter
		int max = 0;
		for (int i = 0; i < speechSignal.length; i++) {
			int sum = 0;
			for (int j = i - LOWPASS_N; j <= i + LOWPASS_N; j++) {
				if (j >= speechSignal.length || j < 0) {
					continue;
				}
				sum += (int) Math.abs(speechSignal[j]);
			}
			int avg = sum/(LOWPASS_N-1);
			if (max < avg) {
				max = avg;
			}
			speechEnvelope[i] = avg;
		}
		
		// find the cross-threshold points and take first two
		final int threshold = (int) (max*SILENCE_THRESHOLD_RATIO);
		List<Integer> crossThresholdPoints = new ArrayList<>();
		for (int i = 0; i < speechEnvelope.length; i++) {
			if (speechEnvelope[i] >= threshold && (i == 0 || speechEnvelope[i-1] < threshold
					|| i == speechEnvelope.length - 1 || speechEnvelope[i+1] < threshold)) {
				crossThresholdPoints.add(i);
			}
		}
		
		int startOfSpeech = crossThresholdPoints.get(0);
		int endOfSpeech = crossThresholdPoints.get(crossThresholdPoints.size() - 1);
		int[] speech = new int[endOfSpeech-startOfSpeech+1];
		for (int i = 0; i < speech.length; i++) {
			speech[i] = speechSignal[startOfSpeech+i];
		}
		return speech;
	}
	
}
