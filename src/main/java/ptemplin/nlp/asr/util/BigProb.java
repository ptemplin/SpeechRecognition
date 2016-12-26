package ptemplin.nlp.asr.util;

import java.math.BigDecimal;

public class BigProb {
	
	// only a single digit
	private double significand;
	private int exponent;

	public BigProb(int exponent) {
		significand = 1;
		this.exponent = exponent;
	}
	
	public BigProb(double significand, int exponent) {
		this.significand = significand;
		this.exponent = exponent;
	}
	
	public BigProb(BigDecimal decimal) {
		this.significand = getSignificand(decimal);
		this.exponent = getExponent(decimal);
	}
	
	public BigProb multiply(BigDecimal decimal) {
		double significand = getSignificand(decimal);
		int exponent = getExponent(decimal);
		double newSig = significand*this.significand;
		int newExp = this.exponent + exponent;
		if (newSig >= 10) {
			newSig /= 10;
			newExp++;
		}
		
		return new BigProb(newSig, newExp);
	}
	
	public BigProb multiply(BigProb other) {
		double newSig = other.significand*this.significand;
		int newExp = this.exponent + other.exponent;
		if (newSig >= 10) {
			newSig /= 10;
			newExp++;
		}
		return new BigProb(newSig, newExp);
	}
	
	public BigProb multiply(double dec) {
		BigDecimal decimal = new BigDecimal(dec);
		return multiply(decimal);
	}
	
	public BigProb divideBy(BigProb other) {
		double newSig = this.significand/other.significand;
		int newExp = this.exponent - other.exponent;
		if (newSig < 1) {
			newExp--;
			newSig *= 10;
		}
		return new BigProb(newSig, newExp);
	}
	
	public BigProb add(BigProb other) {
		double newSig;
		int newExp;
		if (exponent >= other.exponent) {
			newSig = significand + other.significand*Math.pow(10, other.exponent - exponent);
			newExp = exponent;
		} else {
			newSig = other.significand + significand*Math.pow(10, exponent - other.exponent);
			newExp = other.exponent;
		}
		if (newSig >= 10) {
			newSig -= 10;
			newExp++;
		}
		return new BigProb(newSig, newExp);
	}
	
	public double toDouble() {
		return significand*Math.pow(10, exponent);
	}
	
	@Override
	public String toString() {
		return significand + "x10^" + exponent;
	}
	
	
	private static double getSignificand(BigDecimal decimal) {
		return decimal.scaleByPowerOfTen(decimal.scale() - decimal.precision() + 1).doubleValue();
	}
	
	private static int getExponent(BigDecimal decimal) {
		return decimal.precision() - decimal.scale() - 1;
	}
	
	// Test harness
	public static void main(String[] args) {
		// Test #1: Multiplication
		BigProb prob = new BigProb(9,1).multiply(new BigDecimal("0.04697949699493"));
		System.out.println(prob.significand + "x10^" + prob.exponent);
	}
	
}
