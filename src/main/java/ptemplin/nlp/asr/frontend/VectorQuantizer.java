package ptemplin.nlp.asr.frontend;

import java.util.ArrayList;
import java.util.List;

import ptemplin.nlp.asr.clustering.KMeans;

public class VectorQuantizer {

	private static final int CODEBOOK_SIZE = 10;
	
	private List<double[]> codebook = new ArrayList<>();
	
	/**
	 * Initialize the codebook with the training data and
	 * @param trainingData
	 */
	public VectorQuantizer(List<int[]> trainingData) {
		trainCodebook(trainingData);
	}
	
	/**
	 * Train the codebook using K-Means clustering, assigning an index
	 * the the center of each cluster.
	 * @param trainingData to use to train
	 */
	private void trainCodebook(List<int[]> trainingData) {
		codebook = KMeans.cluster(trainingData, CODEBOOK_SIZE);
	}
	
	/**
	 * Employs a nearest-neighbour method for quantizing the observation against
	 * the codebook vectors.
	 * @param observation to be quantized
	 * @return the quantization of the observation as an index in the codebook
	 */
	public int quantizeObservation(int[] observation) {
		int nearestIndex = 0;
		double closestDistance = Integer.MAX_VALUE;
		for (int i = 0; i < codebook.size(); i++) {
			double distance = KMeans.getEuclideanDistance(observation, codebook.get(i));
			if (distance < closestDistance) {
				nearestIndex = i;
				closestDistance = distance;
			}
		}
		return nearestIndex;
	}
	
	public int quantizeObservation(double[] observation) {
		int[] obs = new int[observation.length];
		for (int i = 0; i < observation.length; i++) {
			obs[i] = (int) observation[i];
		}
		return quantizeObservation(obs);
	}
	
	public List<double[]> getCodebook() {
		return codebook;
	}
	
}
