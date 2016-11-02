package ptemplin.nlp.asr.util;

import java.math.BigDecimal;

public class BigProbQuick {
	
	// only a single digit
	private int significand;
	private int exponent;

	public BigProbQuick(int exponent) {
		significand = 1;
		this.exponent = exponent;
	}
	
	public BigProbQuick(int significand, int exponent) {
		this.significand = significand;
		this.exponent = exponent;
	}
	
	public BigProbQuick(BigDecimal decimal) {
		this.significand = getSignificand(decimal);
		this.exponent = getExponent(decimal);
	}
	
	public BigProbQuick multiply(BigDecimal decimal) {
		int significand = getSignificand(decimal);
		int exponent = getExponent(decimal);
		int newSig = significand*this.significand;
		int newExp = this.exponent + exponent;
		if (newSig >= 10) {
			newSig /= 10;
			newExp++;
		}
		return new BigProbQuick(newSig, newExp);
	}
	
	public BigProbQuick multiply(BigProbQuick other) {
		int newSig = other.significand*this.significand;
		int newExp = this.exponent + other.exponent;
		if (newSig >= 10) {
			newSig /= 10;
			newExp++;
		}
		return new BigProbQuick(newSig, newExp);
	}
	
	public BigProbQuick divideBy(BigProbQuick other) {
		int newSig = (this.significand*10)/other.significand;
		int newExp = this.exponent - other.exponent;
		if (newSig < 10) {
			newExp--;
		} else {
			newSig /= 10;
		}
		return new BigProbQuick(newSig, newExp);
	}
	
	public BigProbQuick add(BigProbQuick other) {
		int newSig = significand;
		if (exponent == other.exponent) {
			newSig += other.significand;
		}
		int newExp = exponent;
		if (newSig >= 10) {
			newSig -= 10;
			newExp++;
		}
		return new BigProbQuick(newSig, newExp);
	}
	
	public double toDouble() {
		return significand*Math.pow(10, exponent);
	}
	
	@Override
	public String toString() {
		return significand + "x10^" + exponent;
	}
	
	
	private static int getSignificand(BigDecimal decimal) {
		return decimal.scaleByPowerOfTen(decimal.scale() - decimal.precision() + 1).intValue();
	}
	
	private static int getExponent(BigDecimal decimal) {
		return decimal.precision() - decimal.scale() - 1;
	}
	
	// Test harness
	public static void main(String[] args) {
		// Test #1: Multiplication
		BigProbQuick prob = new BigProbQuick(9,1).multiply(new BigDecimal("0.04697949699493"));
		System.out.println(prob.significand + "x10^" + prob.exponent);
	}
	
}
