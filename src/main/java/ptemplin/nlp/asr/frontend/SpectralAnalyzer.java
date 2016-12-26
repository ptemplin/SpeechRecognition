package ptemplin.nlp.asr.frontend;

import java.util.List;

/**
 * The high-level interface for computing spectral feature vectors from audio data.
 */
public interface SpectralAnalyzer {

    /**
     * Computes the spectral feature vectors from the given raw audio sample.
     *
     * @param sample of raw audio
     * @return the spectral feature vectors computed from this sample
     */
    List<int[]> computeFeatureVectors(int[] sample);

}
