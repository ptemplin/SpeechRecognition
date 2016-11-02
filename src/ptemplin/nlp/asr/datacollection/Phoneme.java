package ptemplin.nlp.asr.datacollection;

/**
 * Phonemes of the English language and whether or not they are a vowel.
 */
public enum Phoneme {

		AA(true),
		AE(true),
		AH(true),
		AO(true),
		AW(true),
		AY(true),
		B(false),
		CH(false),
		D(false),
		DH(false),
		EH(true),
		ER(false),
		EY(true),
		F(false),
		G(false),
		HH(false),
		IH(true),
		IY(true),
		JH(false),
		K(true),
		L(false),
		M(false),
		N(false),
		NG(true),
		OW(true),
		OY(true),
		P(false),
		R(false),
		S(false),
		SH(false),
		T(false),
		TH(false),
		UH(true),
		UW(true),
		V(false),
		W(false),
		Y(false),
		Z(false),
		ZH(false);
		
		private final boolean vowel;
		
		private Phoneme(boolean vowel) {
			this.vowel = vowel;
		}
		
		public boolean isVowel() { return vowel; }
	
}
