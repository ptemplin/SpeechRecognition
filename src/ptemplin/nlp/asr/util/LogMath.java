package ptemplin.nlp.asr.util;

import java.math.BigDecimal;

public class LogMath {

    private BigDecimal significand;

    public LogMath(double significand) {
        this.significand = new BigDecimal(Math.log(significand));
    }

    public LogMath(BigDecimal logSignificand) {
        this.significand = logSignificand;
    }

    public LogMath multiply(LogMath other) {
        return new LogMath(this.significand.add(other.significand));
    }

    public LogMath divide(LogMath other) {
        return new LogMath(this.significand.subtract(other.significand));
    }

    @Override
    public String toString() {
        return "e^" + significand.toString();
    }

    // Test harness
    public static void main(String[] args) {
        // Test #1: Multiplication
        LogMath prob = new LogMath(10.8712124).multiply(new LogMath(0.04697949699493));
        System.out.println(prob);
    }

}
