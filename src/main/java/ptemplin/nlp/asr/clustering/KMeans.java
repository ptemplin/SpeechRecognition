package ptemplin.nlp.asr.clustering;

import java.util.ArrayList;
import java.util.List;

public class KMeans {
	
	private static final int CLUSTER_STEPS = 500;
	
	public static List<double[]> cluster(List<int[]> data, int k) {
		// initialize means to k random points in the dataset
		List<double[]> means = new ArrayList<>();
		for (int i = 0; i < k; i++) {
			means.add(intArrToDoubleArr(data.get((int) (Math.random()*data.size()))));
		}
		// perform the clustering
		int stepNum = 1;
		while (stepNum <= CLUSTER_STEPS) {
			stepNum++;
			// assignment
			List<List<int[]>> clusters = new ArrayList<>();
			for (int i = 0; i < k; i++) {
				clusters.add(new ArrayList<int[]>());
			}
			for (int[] dataPiece : data) {
				int bestCluster = 0;
				double shortestDistance = Integer.MAX_VALUE;
				for (int i = 0; i < means.size(); i++) {
					double distance = getEuclideanDistance(dataPiece, means.get(i));
					if (distance < shortestDistance) {
						shortestDistance = distance;
						bestCluster = i;
					}
				}
				clusters.get(bestCluster).add(dataPiece);
			}
			// update means
			for (int i = 0; i < means.size(); i++) {
				List<int[]> cluster = clusters.get(i);
				double[] mean = means.get(i);
				int dimensionality = mean.length;
				for (int dimension = 0; dimension < dimensionality; dimension++) {
					int sum = 0;
					for (int dataPoint = 0; dataPoint < cluster.size(); dataPoint++) {
						sum += cluster.get(dataPoint)[dimension];
					}
					if (!cluster.isEmpty()) {
						mean[dimension] = sum / (double) cluster.size();
					}
				}
			}
		}
		return means;
	}
	
	public static double getEuclideanDistance(int[] dataPiece, double[] mean) {
		double sum = 0;
		for (int i = 0; i < dataPiece.length; i++) {
			sum += Math.pow(dataPiece[i]-mean[i], 2);
		}
		return Math.sqrt(sum);
	}
	
	private static double[] intArrToDoubleArr(int[] intArr) {
		double[] doubleArr = new double[intArr.length];
		for (int i=0;i<intArr.length;i++) {
			doubleArr[i] = intArr[i];
		}
		return doubleArr;
	}
	
}
