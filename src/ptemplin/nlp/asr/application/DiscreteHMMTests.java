package ptemplin.nlp.asr.application;

import java.util.ArrayList;
import java.util.List;

import ptemplin.nlp.asr.acoustic.ContinuousHMM;
import ptemplin.nlp.asr.acoustic.DiscreteHMM;
import ptemplin.nlp.asr.acoustic.HiddenMarkovModel;
import ptemplin.nlp.asr.frontend.FeatureAnalyzer;
import ptemplin.nlp.asr.frontend.LincolnFrontend;
import ptemplin.nlp.asr.frontend.SilenceProcessor;
import ptemplin.nlp.asr.frontend.VectorQuantizer;
import ptemplin.nlp.asr.io.SpeechFileReader;
import ptemplin.nlp.asr.io.SpeechSample;
import ptemplin.nlp.asr.util.BigProb;

public class DiscreteHMMTests {
	
	private static final LincolnFrontend analyzer = new LincolnFrontend();
	private static final SpeechFileReader reader = new SpeechFileReader();
	
	private static final String[] vocabulary = {"one", "two", "three"};
	private static final int NUM_TRAINING_SAMPLES = 10;

	public static void main(String[] args) throws Exception {
		trainFullModelAndEvaluateSamples();
	}
	
	// Test training analyzed sample against model
	public static void initHmmAndTrain() throws Exception {
		// 1. Create a vector quantizer from all training data
		System.out.println("Initializing VQ...");
		List<int[]> trainingObservationVectors = new ArrayList<>();
		for (String word : vocabulary) {
			for (int i = 1; i <= NUM_TRAINING_SAMPLES; i++) {
				SpeechSample sample = reader.readSpeechFileForTrainingData(word, i);
				int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
				trainingObservationVectors.addAll(analyzer.computeObservationVectors(trimmedSpeech));
			}
		}
		VectorQuantizer vq = new VectorQuantizer(trainingObservationVectors);
		
		// 2. Initialize a discrete HMM with the VQ
		System.out.println("Initializing HMM");
		HiddenMarkovModel acousticModel = new DiscreteHMM(vq);

		// 3. Extracting features from test data and test
		System.out.println("Extracting features from test data");
		SpeechSample sample = reader.readSpeechFileForTrainingData("one", 1);
		int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
		List<int[]> observationSeq = analyzer.computeObservationVectors(trimmedSpeech);
			
		System.out.println("Testing data against default model");
		BigProb likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M) = " + likelihood);
			
		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		
		System.out.println("Testing data against 1xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M') = " + likelihood);
		
		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		
		System.out.println("Testing data against 2xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M'') = " + likelihood);
		
		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		
		System.out.println("Testing data against 3xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M''') = " + likelihood);
		
		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		
		System.out.println("Testing data against 4xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M'4) = " + likelihood);
		
		System.out.println("Training model based on data");
		acousticModel.train(observationSeq);
		
		System.out.println("Testing data against 5xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq);
		System.out.println("P(O|M'5) = " + likelihood);
	}
	
	public static void trainFullModelAndEvaluateSamples() throws Exception {
		String trainingWord = "one";
		
		// 1. Create a vector quantizer from all training data
		System.out.println("Initializing VQ...");
		List<int[]> trainingObservationVectors = new ArrayList<>();
		for (String word : vocabulary) {
			for (int i = 1; i <= NUM_TRAINING_SAMPLES; i++) {
				SpeechSample sample = reader.readSpeechFileForTrainingData(word, i);
				int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
				trainingObservationVectors.addAll(analyzer.computeObservationVectors(trimmedSpeech));
			}
		}
		VectorQuantizer vq = new VectorQuantizer(trainingObservationVectors);
			
		// 2. Initialize a discrete HMM with the VQ
		System.out.println("Initializing HMM");
		HiddenMarkovModel acousticModel = new DiscreteHMM(vq);

		System.out.println("Training on single word...");
		for (int i = 1; i <= 10; i++) {
			SpeechSample sample = reader.readSpeechFileForTrainingData(trainingWord, i);
			int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
			List<int[]> observationSeq = analyzer.computeObservationVectors(trimmedSpeech);
			acousticModel.train(observationSeq);
		}
		
		System.out.println("Testing all vocabulary words against model...");
		for (String word : vocabulary) {
			if (word.equals(trainingWord)) {
				System.out.println("Training word (should be high)");
			} else { System.out.println("Non-trained word (should be low)"); }
			for (int i = 1; i <= 10; i++) {
				SpeechSample sample = reader.readSpeechFileForTrainingData(word, i);
				int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
				List<int[]> observationSeq = analyzer.computeObservationVectors(trimmedSpeech);
				BigProb likelihood = acousticModel.evaluateObservation(observationSeq);
				System.out.println("Word: " + word + " Sample: " + i + " P(O|M) = " + likelihood);
			}
		}
	}
	
	public static void trainAndEvaluate3Samples() throws Exception {
		// 1. Create a vector quantizer from all training data
				System.out.println("Initializing VQ...");
				List<int[]> trainingObservationVectors = new ArrayList<>();
				for (String word : vocabulary) {
					for (int i = 1; i <= NUM_TRAINING_SAMPLES; i++) {
						SpeechSample sample = reader.readSpeechFileForTrainingData(word, i);
						int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
						trainingObservationVectors.addAll(analyzer.computeObservationVectors(trimmedSpeech));
					}
				}
				VectorQuantizer vq = new VectorQuantizer(trainingObservationVectors);
				
				// 2. Initialize a discrete HMM with the VQ
				System.out.println("Initializing HMM");
				HiddenMarkovModel acousticModel = new DiscreteHMM(vq);

		System.out.println("Extracting features from test data");
		SpeechSample sample1 = reader.readSpeechFileForTrainingData("one", 1);
		int[] trimmedSpeech1 = SilenceProcessor.trimSilence(sample1.getDataAsInts());
		List<int[]> observationSeq1 = analyzer.computeObservationVectors(trimmedSpeech1);
		SpeechSample sample2 = reader.readSpeechFileForTrainingData("one", 2);
		int[] trimmedSpeech2 = SilenceProcessor.trimSilence(sample2.getDataAsInts());
		List<int[]> observationSeq2 = analyzer.computeObservationVectors(trimmedSpeech2);
		SpeechSample sample3 = reader.readSpeechFileForTrainingData("one", 3);
		int[] trimmedSpeech3 = SilenceProcessor.trimSilence(sample3.getDataAsInts());
		List<int[]> observationSeq3 = analyzer.computeObservationVectors(trimmedSpeech3);
		SpeechSample sample4 = reader.readSpeechFileForTrainingData("two", 5);
		int[] trimmedSpeech4 = SilenceProcessor.trimSilence(sample4.getDataAsInts());
		List<int[]> observationSeq4 = analyzer.computeObservationVectors(trimmedSpeech4);
		

		System.out.println("Testing data against default model");
		BigProb likelihood = acousticModel.evaluateObservation(observationSeq1);
		System.out.println("P(O_1|M) = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq2);
		System.out.println("P(O_2|M) = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq3);
		System.out.println("P(O_3|M) = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq4);
		System.out.println("P(O_4|M) = " + likelihood + "    (Not a one)");

		System.out.println("Training model based on sample data #1");
		acousticModel.train(observationSeq1);

		System.out.println("Testing data against 1xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq1);
		System.out.println("P(O_1|M') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq2);
		System.out.println("P(O_2|M') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq3);
		System.out.println("P(O_3|M') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq4);
		System.out.println("P(O_4|M) = " + likelihood + "    (Not a one)");
		
		System.out.println("Training model based on sample data #2");
		acousticModel.train(observationSeq2);

		System.out.println("Testing data against 2xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq1);
		System.out.println("P(O_1|M'') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq2);
		System.out.println("P(O_2|M'') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq3);
		System.out.println("P(O_3|M'') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq4);
		System.out.println("P(O_4|M) = " + likelihood + "    (Not a one)");
		
		System.out.println("Training model based on sample data #3");
		acousticModel.train(observationSeq3);

		System.out.println("Testing data against 3xtrained model");
		likelihood = acousticModel.evaluateObservation(observationSeq1);
		System.out.println("P(O_1|M''') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq2);
		System.out.println("P(O_2|M''') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq3);
		System.out.println("P(O_3|M''') = " + likelihood);
		likelihood = acousticModel.evaluateObservation(observationSeq4);
		System.out.println("P(O_4|M) = " + likelihood + "    (Not a one)");
	}
	
	// Test feature analysis, vector quantization for training data
		public static void buildCodebookAndTest() throws Exception {
			// Part One: Train the vector quantizer against training data
			// create a vector quantizer from all training data
			System.out.println("Training...");
			List<int[]> trainingObservationVectors = new ArrayList<>();
			for (String word : vocabulary) {
				for (int i = 1; i <= NUM_TRAINING_SAMPLES; i++) {
					SpeechSample sample = reader.readSpeechFileForTrainingData(word, i);
					int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
					trainingObservationVectors.addAll(analyzer.computeObservationVectors(trimmedSpeech));
				}
			}
			VectorQuantizer vq = new VectorQuantizer(trainingObservationVectors);
			// Part Two: Test the vector quantizer against real speech
			// get the observation vectors for a word in the vocab
			System.out.println("Testing");
			testDataAgainstCodebook(vq, "one", 1);
			testDataAgainstCodebook(vq, "one", 2);
			testDataAgainstCodebook(vq, "one", 3);
			testDataAgainstCodebook(vq, "two", 1);
			testDataAgainstCodebook(vq, "two", 2);
			testDataAgainstCodebook(vq, "two", 3);
			testDataAgainstCodebook(vq, "three", 1);
			testDataAgainstCodebook(vq, "three", 2);
			testDataAgainstCodebook(vq, "three", 3);
		}
		
		private static void testDataAgainstCodebook(VectorQuantizer vq, String word, int n) throws Exception {
			List<int[]> testObservationVectors = new ArrayList<>();
			SpeechSample sample = reader.readSpeechFileForTrainingData(word, n);
			int[] trimmedSpeech = SilenceProcessor.trimSilence(sample.getDataAsInts());
			testObservationVectors.addAll(analyzer.computeObservationVectors(trimmedSpeech));
			// classify each vector and print the sequence
			System.out.println(word + " #" + n);
			for (int[] observationVector : testObservationVectors) {
				int code = vq.quantizeObservation(observationVector);
				System.out.print(code + " ");
			}
			System.out.println();
		}
	
	private static void printObservationVector(List<int[]> observationSequence) {
		for (int i = 0; i < observationSequence.get(0).length; i++) {
			for (int j = 0; j < observationSequence.size(); j++) {
				System.out.format("%-4d  ", observationSequence.get(j)[i]);
			}
			System.out.println();
		}
	}
	
}
