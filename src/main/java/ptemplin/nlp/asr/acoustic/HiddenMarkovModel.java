package ptemplin.nlp.asr.acoustic;

import static ptemplin.nlp.asr.frontend.FeatureAnalyzer.NUM_MFCC_COMPONENTS;

import java.math.BigDecimal;
import java.util.List;

import ptemplin.nlp.asr.util.*;

/**
 * Defines a HMM with linear topology, initialized with a default flat start.
 */
public abstract class HiddenMarkovModel {
	
	protected static final int DEFAULT_NUM_STATES = 6;
	protected static final int DEFAULT_OBS_SIZE = NUM_MFCC_COMPONENTS;
	
	protected static final double MIN_LOG_TRANSITION_PROB = -15000;
	
	protected final int numStates;
	protected final int observationSize;
	
	protected final double[][] transitionProbs;
	
	protected double[][] currentObservationSeq;

	protected final LogMath logMath;
	
	HiddenMarkovModel() {
		numStates = DEFAULT_NUM_STATES;
		observationSize = DEFAULT_OBS_SIZE;
		transitionProbs = new double[numStates][numStates];
		logMath = LogMath.getLogMath();
	}
	
	public double[][] getTransitionProbs() { return transitionProbs; }
	
	public double evaluateObservation(List<int[]> observationSeq) {
		double[][] obsSeq = new double[observationSeq.size()][observationSize];
		for (int i = 0; i < observationSeq.size(); i++) {
			for (int j = 0; j < observationSize; j++) {
				obsSeq[i][j] = observationSeq.get(i)[j];
			}
		}
		return evaluateObservation(obsSeq);
	}
	
	public double evaluateObservation(double[][] observationSeq) {
		currentObservationSeq = observationSeq;
		double[][] logForwardProbs = computeForwardProbabilities(observationSeq);
        System.out.println("e^" + logMath.logToLn(logForwardProbs[9][5]));
		return logMath.logToLinear(logForwardProbs[observationSeq.length][numStates-1]);
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
		double[][] logForwardProbs = computeForwardProbabilities(observationSeq);
		double[][] logBackwardProbs = computeBackwardProbabilities(observationSeq);
		double[][][] logArcProbs = computeArcProbabilities(logForwardProbs, logBackwardProbs, observationSeq);
		updateTransitionProbabilities(logArcProbs);
		updateOutputParameters(logArcProbs, observationSeq);
	}
	
	private double[][] computeForwardProbabilities(double[][] observationSeq) {
		final int totalTime = observationSeq.length;
		double[][] logForwardProbs = new double[totalTime + 1][numStates];
		// initialize the base case at state #1
		logForwardProbs[0][0] = LogMath.LOG_ONE;
		// for every timestep t
		for (int t = 1; t < totalTime + 1; t++) {
		// for (int t = 1; t < 10; t++) {
			// for each state s
			for (int s = 0; s < numStates; s++) {
				double logSum = Double.NEGATIVE_INFINITY;
				// for each state that leads to this one i
				if (s > 0 && t - s >= 0) {
					logSum = logForwardProbs[t-1][s-1]
							+ logMath.linearToLog(transitionProbs[s-1][s])
							+ computeOutputProbability(s-1, s, t-1);
				}
				if (t - s >= 1) {
					double logSecondComponent = logForwardProbs[t-1][s]
							+ logMath.linearToLog(transitionProbs[s][s])
							+ computeOutputProbability(s, s, t-1);
					if (logSum == Double.NEGATIVE_INFINITY) {
						logSum = logSecondComponent;
					} else {
						logSum = logMath.addAsLinear(logSum, logSecondComponent);
					}
				}
				//System.out.println("Forward probability (" + t + "," + s + "): " + logSum);
				logForwardProbs[t][s] = logSum;
			}
		}
		return logForwardProbs;
	}
	
	private double[][] computeBackwardProbabilities(double[][] observationSeq) {
		final int totalTime = observationSeq.length;
		double[][] logBackwardProbs = new double[totalTime + 1][numStates];
		// initialize the base case at state #1
		logBackwardProbs[totalTime][numStates-1] = LogMath.LOG_ONE;
		// for every timestep t
		for (int t = totalTime-1; t >= 0; t--) {
			// for each state s
			for (int s = numStates-1; s >= 0; s--) {
				double logSum = Double.NEGATIVE_INFINITY;
				// for each state that leads to this one i
				if (s < numStates-1 && t + ((numStates - 1) - s) <= totalTime) {
					logSum = logBackwardProbs[t+1][s+1]
							+ logMath.linearToLog(transitionProbs[s][s+1])
							+ computeOutputProbability(s, s+1, t);
				}
				if (t + ((numStates - 1) - s) <= totalTime - 1) {
					double logSecondComponent = logBackwardProbs[t+1][s]
							+ logMath.linearToLog(transitionProbs[s][s])
							+ computeOutputProbability(s, s, t);
					if (logSum == Double.NEGATIVE_INFINITY) {
						logSum = logSecondComponent;
					} else {
						logSum = logMath.addAsLinear(logSum, logSecondComponent);
					}
				}
				//System.out.println("Backward probability (" + t + "," + s + "): " + logSum);
				logBackwardProbs[t][s] = logSum;
			}
		}
		return logBackwardProbs;
	}
	
	private double[][][] computeArcProbabilities(double[][] logForwardProbs, double[][] logBackwardProbs, double[][] observationSeq) {
		final int totalTime = logForwardProbs.length;
        final double logModelProbability = logForwardProbs[totalTime-1][numStates-1];
		double[][][] logArcProbs = new double[numStates][numStates][totalTime-1];
		// for every timestep t
		for (int t = 0; t < totalTime - 1; t++) {
			// for each state s
			for (int s = 0; s < numStates; s++) {
				// for each state that this one leads to
				if (logForwardProbs[t][s] != 0.d && logBackwardProbs[t+1][s] != 0.d) {
					double logSelfArcProb = logForwardProbs[t][s]
							+ logMath.linearToLog(transitionProbs[s][s])
							+ computeOutputProbability(s, s, t)
							+ logBackwardProbs[t+1][s]
                            - logModelProbability;
					//System.out.println("Arc probability (" + s + ", " + s + ", " + t + "): " + logSelfArcProb);
					logArcProbs[s][s][t] = logSelfArcProb;
				}
				if (s < numStates-1
						&& logForwardProbs[t][s+1] != 0.d
						&& logBackwardProbs[t+1][s+1] != 0.d) {
					double logNextStateArcProb = logForwardProbs[t][s+1]
							+ logMath.linearToLog(transitionProbs[s][s+1])
							+ computeOutputProbability(s, s+1, t)
							+ logBackwardProbs[t+1][s+1]
                            - logModelProbability;
					//System.out.println("Arc probability (" + s + ", " + (s+1) + ", " + t + "): " + logNextStateArcProb);
					logArcProbs[s][s+1][t] = logNextStateArcProb;
				}
			}
		}
		return logArcProbs;
	}
	
	private void updateTransitionProbabilities(double[][][] logArcProbs) {
		final int totalTime = logArcProbs[0][0].length;
		for (int i = 0; i < numStates; i++) {
			for (int j = i; j <= i+1 && j < numStates; j++) {
				double logNumerator = Double.NEGATIVE_INFINITY;
				for (int t = 0; t < totalTime; t++) {
					if (logArcProbs[i][j][t] != 0.d) {
						if (logNumerator == Double.NEGATIVE_INFINITY) {
							logNumerator = logArcProbs[i][j][t];
						} else {
							logNumerator = logMath.addAsLinear(logNumerator, logArcProbs[i][j][t]);
						}
					}
				}
				double logDenominator = Double.NEGATIVE_INFINITY;
				for (int n = 0; n < numStates; n++) {
					for (int t = 0; t < totalTime; t++) {
						if (logArcProbs[i][n][t] != 0.d) {
							if (logDenominator == Double.NEGATIVE_INFINITY) {
								logDenominator = logArcProbs[i][n][t];
							} else {
								logDenominator = logMath.addAsLinear(logDenominator, logArcProbs[i][n][t]);
							}
						}
					}
				}
				//System.out.println(logNumerator + "/" + logDenominator);
				double result = logNumerator - logDenominator;
                //System.out.println("=" + result);
				//System.out.print("Old a(" + i + "," + j + "): " + transitionProbs[i][j] + "   ");
				if (result < MIN_LOG_TRANSITION_PROB) {
					result = MIN_LOG_TRANSITION_PROB;
				}
				transitionProbs[i][j] = logMath.logToLinear(result);
				//System.out.println("New a(" + i + "," + j + "): " + transitionProbs[i][j] + "   ");
			}
			normalizeTransitionProbabilities(i);
		}
	}

	/**
	 * Normalizes the transition probabilities out of a given state such that they add to 1.0.
	 *
	 * @param startState the index of the state with outgoing arcs
	 */
	private void normalizeTransitionProbabilities(int startState) {
        if (startState != numStates - 1) {
            double arcSum = transitionProbs[startState][startState] + transitionProbs[startState][startState + 1];
            transitionProbs[startState][startState] /= arcSum;
            transitionProbs[startState][startState + 1] /= arcSum;
        }
    }
	
	protected abstract void updateOutputParameters(double[][][] arcProbs, double[][] observationSeq);
	
	protected abstract double computeOutputProbability(int state1, int state2, int observationNum);

}
