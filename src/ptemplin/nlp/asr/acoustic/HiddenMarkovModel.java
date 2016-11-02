package ptemplin.nlp.asr.acoustic;

import static ptemplin.nlp.asr.frontend.FeatureAnalyzer.NUM_MFCC_COMPONENTS;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;

import org.jlinalg.Matrix;
import org.jlinalg.bigdecimalwrapper.BigDecimalWrapper;
import org.jlinalg.bigdecimalwrapper.BigDecimalWrapperFactory;

import ptemplin.nlp.asr.util.BigProb;
import ptemplin.nlp.asr.util.Gaussian;
import ptemplin.nlp.asr.util.PrecisionMathUtils;
import ptemplin.nlp.asr.util.ValidationTools;

/**
 * Defines a HMM with linear topology, initialized with a default flat start.
 */
public abstract class HiddenMarkovModel {
	
	protected static final int DEFAULT_NUM_STATES = 6;
	protected static final int DEFAULT_OBS_SIZE = NUM_MFCC_COMPONENTS;
	
	protected static final double MIN_TRANSITION_PROB = 0.00001;
	
	protected final int numStates;
	protected final int observationSize;
	
	protected final double[][] transitionProbs;
	
	protected double[][] currentObservationSeq;
	
	public HiddenMarkovModel() {
		numStates = DEFAULT_NUM_STATES;
		observationSize = DEFAULT_OBS_SIZE;
		transitionProbs = new double[numStates][numStates];
	}
	
	public double[][] getTransitionProbs() { return transitionProbs; }
	
	public BigProb evaluateObservation(List<int[]> observationSeq) {
		double[][] obsSeq = new double[observationSeq.size()][observationSize];
		for (int i = 0; i < observationSeq.size(); i++) {
			for (int j = 0; j < observationSize; j++) {
				obsSeq[i][j] = observationSeq.get(i)[j];
			}
		}
		return evaluateObservation(obsSeq);
	}
	
	public BigProb evaluateObservation(double[][] observationSeq) {
		currentObservationSeq = observationSeq;
		BigProb[][] forwardProbs = computeForwardProbabilities(observationSeq);
		return forwardProbs[observationSeq.length][numStates-1];
	}
	
	public void train(List<int[]> observationSeq) {
		double[][] obsSeq = new double[observationSeq.size()][observationSize];
		for (int i = 0; i < observationSeq.size(); i++) {
			for (int j = 0; j < observationSize; j++) {
				obsSeq[i][j] = observationSeq.get(i)[j];
			}
		}
		train(obsSeq);
	}
	
	public void train(double[][] observationSeq) {
		currentObservationSeq = observationSeq;
		BigProb[][] forwardProbs = computeForwardProbabilities(observationSeq);
		BigProb[][] backwardProbs = computeBackwardProbabilities(observationSeq);
		BigProb[][][] arcProbs = computeArcProbabilities(forwardProbs, backwardProbs, observationSeq);
		updateTransitionProbabilities(arcProbs);
		updateOutputParameters(arcProbs, observationSeq);
	}
	
	private BigProb[][] computeForwardProbabilities(double[][] observationSeq) {
		final int totalTime = observationSeq.length;
		BigProb[][] forwardProbs = new BigProb[totalTime + 1][numStates];
		// initialize the base case at state #1
		forwardProbs[0][0] = new BigProb(0);
		// for every timestep t
		for (int t = 1; t < totalTime + 1; t++) {
		// for (int t = 1; t < 10; t++) {
			// for each state s
			for (int s = 0; s < numStates; s++) {
				BigProb sum = null;
				// for each state that leads to this one i
				if (s > 0 && forwardProbs[t-1][s-1] != null) {
					sum = forwardProbs[t-1][s-1]
							.multiply(new BigDecimal(transitionProbs[s-1][s]))
							.multiply(computeOutputProbability(s-1, s, t-1));
				}
				if (forwardProbs[t-1][s] != null) {
					BigProb secondComponent = forwardProbs[t-1][s]
							.multiply(new BigDecimal(transitionProbs[s][s]))
							.multiply(computeOutputProbability(s, s, t-1));
					if (sum == null) {
						sum = secondComponent;
					} else {
						sum = sum.add(secondComponent);
					}
				}
				//System.out.println("Forward probability (" + t + "," + s + "): " + sum);
				forwardProbs[t][s] = sum;
			}
		}
		return forwardProbs;
	}
	
	private BigProb[][] computeBackwardProbabilities(double[][] observationSeq) {
		final int totalTime = observationSeq.length;
		BigProb[][] backwardProbs = new BigProb[totalTime + 1][numStates];
		// initialize the base case at state #1
		backwardProbs[totalTime][numStates-1] = new BigProb(0);
		// for every timestep t
		for (int t = totalTime-1; t >= 0; t--) {
			// for each state s
			for (int s = numStates-1; s >= 0; s--) {
				BigProb sum = null;
				// for each state that leads to this one i
				if (s < numStates-1 && backwardProbs[t+1][s+1] != null) {
					sum = backwardProbs[t+1][s+1]
							.multiply(new BigDecimal(transitionProbs[s][s+1]))
							.multiply(computeOutputProbability(s, s+1, t));
				}
				if (backwardProbs[t+1][s] != null) {
					BigProb secondComponent = backwardProbs[t+1][s]
							.multiply(new BigDecimal(transitionProbs[s][s]))
							.multiply(computeOutputProbability(s, s, t));
					if (sum == null) {
						sum = secondComponent;
					} else {
						sum = sum.add(secondComponent);
					}
				}
				// System.out.println("Backward probability (" + t + "," + s + "): " + sum);
				backwardProbs[t][s] = sum;
			}
		}
		return backwardProbs;
	}
	
	private BigProb[][][] computeArcProbabilities(BigProb[][] forwardProbs, BigProb[][] backwardProbs, double[][] observationSeq) {
		final int totalTime = forwardProbs.length;
		BigProb[][][] arcProbs = new BigProb[numStates][numStates][totalTime-1];
		// for every timestep t
		for (int t = 0; t < totalTime - 1; t++) {
			// for each state s
			for (int s = 0; s < numStates; s++) {
				// for each state that this one leads to
				if (forwardProbs[t][s] != null && backwardProbs[t+1][s] != null) {
					BigProb selfArcProb = forwardProbs[t][s]
							.multiply(new BigDecimal(transitionProbs[s][s]))
							.multiply(computeOutputProbability(s, s, t))
							.multiply(backwardProbs[t+1][s]);
					// System.out.println("Arc probability (" + s + ", " + s + ", " + t + "): " + selfArcProb);
					arcProbs[s][s][t] = selfArcProb;
				}
				if (s < numStates-1 && forwardProbs[t][s+1] != null && backwardProbs[t+1][s+1] != null) {
					BigProb nextStateArcProb = forwardProbs[t][s+1]
							.multiply(new BigDecimal(transitionProbs[s][s+1]))
							.multiply(computeOutputProbability(s, s+1, t))
							.multiply(backwardProbs[t+1][s+1]);
					// System.out.println("Arc probability (" + s + ", " + (s+1) + ", " + t + "): " + nextStateArcProb);
					arcProbs[s][s+1][t] = nextStateArcProb;
				}
			}
		}
		return arcProbs;
	}
	
	public void updateTransitionProbabilities(BigProb[][][] arcProbs) {
		final int totalTime = arcProbs[0][0].length;
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j <= i+1 && j < numStates; j++) {
				BigProb numerator = null;
				for (int t = 0; t < totalTime; t++) {
					if (arcProbs[i][j][t] != null) {
						if (numerator == null) {
							numerator = arcProbs[i][j][t];
						} else {
							numerator = numerator.add(arcProbs[i][j][t]);
						}
					}
				}
				BigProb denominator = null;
				for (int n = 0; n < numStates; n++) {
					for (int t = 0; t < totalTime; t++) {
						if (arcProbs[i][n][t] != null) {
							if (denominator == null) {
								denominator = arcProbs[i][n][t];
							} else {
								denominator = denominator.add(arcProbs[i][n][t]);
							}
						}
					}
				}
				double result = numerator.divideBy(denominator).toDouble();
				// System.out.print("Old a(" + i + "," + j + "): " + transitionProbs[i][j] + "   ");
				if (result < MIN_TRANSITION_PROB) {
					result = MIN_TRANSITION_PROB;
				}
				transitionProbs[i][j] = result;
				// System.out.println("New a(" + i + "," + j + "): " + transitionProbs[i][j] + "   ");
			}
		}
	}
	
	protected abstract void updateOutputParameters(BigProb[][][] arcProbs, double[][] observationSeq);
	
	protected abstract BigDecimal computeOutputProbability(int state1, int state2, int observationNum);

}
