package ptemplin.nlp.asr.frontend;

public interface FilterBank {
	
	public double[] filter(double[] signal, double frequencyMultiplier);
	
	public int getSize();
	
	public int getLow();
	
	public int getHigh();
	
}
