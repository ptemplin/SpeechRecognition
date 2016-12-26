package ptemplin.nlp.asr.frontend.lincoln;

import ptemplin.nlp.asr.frontend.FilterBank;

public class MelFrequencyFilterBank implements FilterBank {
	
	private final int size;
	private final int low;
	private final int high;
	private final int[] frequencyBins;
	
	private static final int DEFAULT_SIZE = 33;
	private static final int DEFAULT_LOW = 0;
	private static final int DEFAULT_HIGH = 2250;
	
	public MelFrequencyFilterBank() {
		this(DEFAULT_SIZE);
	}
	
	public MelFrequencyFilterBank(int size) {
		this(size, DEFAULT_LOW, DEFAULT_HIGH);
	}
	
	public MelFrequencyFilterBank(int size, int low, int high) {
		this.size = size;
		this.low = low;
		this.high = high;
		this.frequencyBins = new int[size];
		int interval = (high-low) / (size - 1);
		frequencyBins[0] = 20;
		for (int i = 1; i < frequencyBins.length; i++) {
			float currentMel = interval*(i);
			frequencyBins[i] = (int) (700*(Math.pow(10, currentMel/2595) - 1));
		}
	}
	
	@Override
	public double[] filter(double[] signal, double freqMultiplier) {
		double[] coefficients = new double[frequencyBins.length];
		// for each frequency bin
		for (int i = 0; i < coefficients.length; i++) {
			float peak = frequencyBins[i];
			float rightEnd;
			if (i == coefficients.length - 1) {
				rightEnd = frequencyBins[i] + (int) (peak-frequencyBins[i-1]);
			} else {
				rightEnd = frequencyBins[i+1];
			}
			float leftEnd;
			if (i == 0) {
				leftEnd = 0;
			} else {
				leftEnd = frequencyBins[i-1];
			}
			float leftSlope = 1/(peak-leftEnd);
			float rightSlope = 1/(peak-rightEnd);
			// the left side of the band filter
			for (int samplei = (int) Math.ceil(leftEnd/freqMultiplier); samplei <= (int) Math.ceil(peak/freqMultiplier); samplei++) {
				coefficients[i] += signal[samplei]*((samplei*freqMultiplier-leftEnd)*leftSlope);
			}
			// the right side of the band filter
			for (int samplei = (int) Math.ceil(peak/freqMultiplier) + 1; samplei <= (int) Math.floor(rightEnd/freqMultiplier); samplei++) {
				coefficients[i] += signal[samplei]*((peak-samplei*freqMultiplier)*rightSlope);
			}
		}	
		return coefficients;
	}
	
	@Override
	public int getSize() {
		return size;
	}
	
	@Override
	public int getLow() {
		return low;
	}
	
	@Override
	public int getHigh() {
		return high;
	}
	
}
