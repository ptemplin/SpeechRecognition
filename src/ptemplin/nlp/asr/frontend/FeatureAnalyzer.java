package ptemplin.nlp.asr.frontend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;

import ptemplin.nlp.asr.io.SpeechFileReader;
import ptemplin.nlp.asr.io.SpeechSample;

public class FeatureAnalyzer {
	
	private static final int SAMPLE_RATE = 12000;
	// 20ms frame length
	private static final int FRAME_LENGTH = 240;
	private static final int PADDED_FRAME_LENGTH = 256;
	private static final float FRAMES_PER_SECOND = SAMPLE_RATE/PADDED_FRAME_LENGTH;
	private static final int PADDING_LENGTH = (PADDED_FRAME_LENGTH - FRAME_LENGTH)/2;
	
	public static final int NUM_MFCC_COMPONENTS = 13;
	
	private static final double[] HAMMING_WINDOW = generateHammingWindow(FRAME_LENGTH);
	
	private final MelFrequencyFilterBank melFilter;
	private final FastFourierTransformer fftExecutor;
	private final FastCosineTransformer fctExecutor;
	
	public FeatureAnalyzer() {
		this.melFilter = new MelFrequencyFilterBank();
		this.fftExecutor = new FastFourierTransformer(DftNormalization.STANDARD);
		this.fctExecutor = new FastCosineTransformer(DctNormalization.STANDARD_DCT_I);
	}
	
	public List<int[]> computeObservationVectors(int[] sample) {
		List<int[]> frameMFCCs = new ArrayList<>();
		int startIndex = 0;
		while (startIndex < sample.length) {
			int[] frame = Arrays.copyOfRange(sample, startIndex, Math.min(startIndex + FRAME_LENGTH, sample.length));
			double[] rawMFCC = computeFrameMFCC(frame);
			// discard first component, round, and truncate
			int[] roundedMFCC = new int[NUM_MFCC_COMPONENTS];
			for (int i = 1; i < roundedMFCC.length; i++) { roundedMFCC[i-1] = (int) rawMFCC[i]; }
			frameMFCCs.add(roundedMFCC);
			// shift frame by length/2
			startIndex += FRAME_LENGTH/2;
		}
		return frameMFCCs;
	}
	
	private double[] computeFrameMFCC(int[] frame) {
		double[] preprocessedFrame = preprocess(frame);
		double[] powerSpectrum = getPowerSpectrum(preprocessedFrame);
		preemphasize(powerSpectrum);
		double[] melFreqSpectrum = melFilter.filter(powerSpectrum, FRAMES_PER_SECOND);
		convertToLogPowerSpectrum(melFreqSpectrum);
		double[] cepstrum = getDiscreteCosineTransform(melFreqSpectrum);
		applyCepstralWeighting(cepstrum);
		return cepstrum;
	}
	
	private static double[] generateHammingWindow(int length) {
		final double ALPHA_FACTOR = 0.54;
		final double BETA_FACTOR = 0.46;
		double[] hammingWindow = new double[length];
		for (int i = 0; i < length; i++) {
			hammingWindow[i] = ALPHA_FACTOR - BETA_FACTOR*Math.cos((2*i*Math.PI)/(length-1));
		}
		return hammingWindow;
	}
	
	private static double[] preprocess(int[] speechData) {
		assert(speechData.length == FRAME_LENGTH);
		double[] windowedData = new double[PADDED_FRAME_LENGTH];
		for (int i = 0; i < speechData.length; i++) {
			windowedData[i + PADDING_LENGTH] = speechData[i]*HAMMING_WINDOW[i];
		}
		return windowedData;
	}
	
	private double[] getPowerSpectrum(double[] data) {
		Complex[] complexSpectrum = fftExecutor.transform(data, TransformType.FORWARD);
		double[] powerSpectrum = new double[complexSpectrum.length];
		for (int i = 0; i < complexSpectrum.length; i++) {
			powerSpectrum[i] = Math.pow(complexSpectrum[i].abs(), 2);
		}
		return powerSpectrum;
	}
	
	private static void preemphasize(double[] data) {
		for (int i = 0; i < data.length - 1; i++) {
			data[i+1] = 0.95*data[i+1] + 0.05*data[i];
		}
	}
	
	private static void convertToLogPowerSpectrum(double[] powerSpectrum) {
		for (int i = 0; i < powerSpectrum.length; i++) {
			powerSpectrum[i] = 10*Math.log10(powerSpectrum[i]);
		}
	}
	
	private double[] getDiscreteCosineTransform(double[] logPowerSpectrum) {
		return fctExecutor.transform(logPowerSpectrum, TransformType.FORWARD);
	}
	
	private void applyCepstralWeighting(double[] cepstrum) {
		for (int i = 1; i <= cepstrum.length; i++) {
			cepstrum[i-1] *= 1 + (NUM_MFCC_COMPONENTS/2)*Math.sin((Math.PI*i)/NUM_MFCC_COMPONENTS);
		}
	}
	
	public static void main(String[] args) throws Exception {
		FeatureAnalyzer analyzer = new FeatureAnalyzer();
		List<int[]> observations = analyzer.computeObservationVectorsForPhone("R");
		for (int[] obs : observations) { System.out.println(Arrays.toString(obs)); }
	}
	
	private List<int[]> computeObservationVectorsForPhone(String phone) throws Exception {
		SpeechFileReader speechReader = new SpeechFileReader();
		SpeechSample fullSample = speechReader.readSpeechFileForPhone(phone);
		int[] trimmedSpeech = SilenceProcessor.trimSilence(fullSample.getDataAsInts());
		return computeObservationVectors(trimmedSpeech);
	}
	
}
