package ptemplin.nlp.asr.util;

import java.math.BigDecimal;

/**
 * Provides utilities methods for computing Gaussian probabilities.
 */
public class Gaussian {
	
	private static final double[] TEST_MEANS = {400, 500, 450, 500, 200, 50, -50, 75, 100, 40, 20, 10, 5};
	private static final double[] TEST_VARIANCES = {200, 200, 200, 100, 30, 20, 20, 30, 50, 30, 50, 10, 10, 10};
	private static final boolean DEBUG = false;

	public static BigDecimal multiGaussian(double[] mean, double[][] variance, double[] obs) {
		final BigDecimal minComponent = new BigDecimal("1E-250");
		BigDecimal prob = new BigDecimal("1");
		for (int i = 0; i < mean.length; i++) {
			double meanDiff = mean[i] - obs[i];
			double exponent = -Math.pow(meanDiff, 2)/(2*variance[i][i]);
			double divisor = 2*variance[i][i]*Math.PI;
			if (divisor < 0.000000000000001) {
				prob = prob.multiply(minComponent);
			} else {
				BigDecimal componentProb = new BigDecimal(Math.exp(exponent)/divisor);
				if (componentProb.compareTo(minComponent) < 0) {
					componentProb = minComponent;
				}
				prob = prob.multiply(componentProb);
			}
		}
		if (DEBUG) {
			System.out.println("Probability: " + PrecisionMathUtils.bigDecimalToString(prob));
		}
		return prob;
	}
	
	public static void main(String[] args) {
		System.out.println("Between 1 and 2 SD^2 from mean");
		multiGaussian(TEST_MEANS, buildVariances(), new double[]{600, 300, 700, 360, 150, 40, -15, 40, 190, -5, 32, 24, -14});
		System.out.println("Between 0 and 1 SD^2 from mean");
		multiGaussian(TEST_MEANS, buildVariances(), new double[]{450, 534, 200, 487, 220, 40, -45, 75, 99, 57, 22, 18, -4});
	}
	
	private static double[][] buildVariances() {
		int len = TEST_VARIANCES.length;
		double[][] variances = new double[len][len];
		for (int i = 0; i < len; i++) {
			variances[i][i] = TEST_VARIANCES[i];
		}
		return variances;
	}
	
}
