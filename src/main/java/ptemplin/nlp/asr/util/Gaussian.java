package ptemplin.nlp.asr.util;

import java.math.BigDecimal;

/**
 * Provides utilities methods for computing Gaussian probabilities.
 */
public class Gaussian {
	
	private static final double[] TEST_MEANS = {400, 500, 450, 500, 200, 50, -50, 75, 100, 40, 20, 10, 5};
	private static final double[] TEST_VARIANCES = {200, 200, 200, 100, 30, 20, 20, 30, 50, 30, 50, 10, 10, 10};
	private static final boolean DEBUG = false;

	private static final BigDecimal MIN_COMPONENT = new BigDecimal("1E-50");

	public static BigDecimal multiGaussian(double[] mean, double[][] variance, double[] obs) {
		BigDecimal prob = new BigDecimal("1");
		for (int i = 0; i < mean.length; i++) {
			double meanDiff = mean[i] - obs[i];
			double exponent = -Math.pow(meanDiff, 2)/(2*variance[i][i]);
			double divisor = 2*variance[i][i]*Math.PI;
			if (divisor < 1E-50) {
				prob = prob.multiply(MIN_COMPONENT);
			} else {
				BigDecimal componentProb;
				try {
					componentProb = new BigDecimal(Math.exp(exponent) / divisor);
				} catch (NumberFormatException ex) {
					System.out.println("Exponent: " + exponent);
					System.out.println("Divisor: " + divisor);
					System.out.println("Mean: " + mean[i]);
					System.out.println("Obs: " + obs[i]);
					System.out.println("Variance: " + variance[i][i]);
					System.out.println("i: " + i);
					prob = prob.multiply(MIN_COMPONENT);
					throw ex;
					// continue;
				}
				if (componentProb.compareTo(MIN_COMPONENT) < 0) {
					componentProb = MIN_COMPONENT;
				}
				prob = prob.multiply(componentProb);
			}
		}
		if (DEBUG) {
			System.out.println("Probability: " + prob);
		}
		return prob;
	}
	
	public static void main(String[] args) {
		System.out.println("Between 1 and 2 SD^2 from mean");
		BigDecimal x = multiGaussian(TEST_MEANS, buildVariances(), new double[]{600, 300, 700, 360, 150, 40, -15, 40, 190, -5, 32, 24, -14});
		System.out.println(x);
		System.out.println("Between 0 and 1 SD^2 from mean");
		x = multiGaussian(TEST_MEANS, buildVariances(), new double[]{450, 534, 200, 487, 220, 40, -45, 75, 99, 57, 22, 18, -4});
		System.out.println(x);
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
