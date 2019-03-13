package ai.eloquent.rephrasing.scripts.idk;

import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.util.ArgumentParser;
import edu.stanford.nlp.util.StringUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GenerateDataset {

  @ArgumentParser.Option(name="in", gloss="The input Turk data file. This should be a csv file",
      required=true)
  public static String csvFile = "rawdata.csv";

  @ArgumentParser.Option(name="out", gloss="The output dataset file. This should be a tsv file",
      required=true)
  public static String outFile = "outputdataset.csv";

  public static String[] questionColumns = {"Input.question_0", "Input.question_1", "Input.question_2", "Input.question_3"};

  public static String[] answerColumns = {"Answer.response_0", "Answer.response_1", "Answer.response_2", "Answer.response_3"};

  private static List<String> splitCSVLine(String csvLine) {
    List<String> out = new ArrayList<String>();
    int index = 0;
    while (index != -1 && index < csvLine.length()) {
      if (csvLine.charAt(index) == '"') {
        int nextIndex = csvLine.indexOf('"', index + 1);
        out.add(csvLine.substring(index + 1, nextIndex));
        index = nextIndex + 1;
      } else if (csvLine.charAt(index) == ',') {
        index += 1;
      } else {
        int nextIndex = csvLine.indexOf(',', index + 1);
        if (nextIndex != -1) {
          out.add(csvLine.substring(index, nextIndex));
          index = nextIndex + 1;
        } else {
          out.add(csvLine.substring(index));
          index = -1;
        }
      }
    }


    return out;
  }

  private static List<String[]> readSentences() {
    List<String[]> sentences = new ArrayList<>();

    BufferedReader br = null;
    String line;
    try {

      br = new BufferedReader(new FileReader(csvFile));

      line = br.readLine();
      if (line == null) {
        return sentences;
      }

      String[] header = new String[0];
      header = splitCSVLine(line).toArray(header);
      //Array of indices of columns where the questions are located
      int[] questionIndices = new int[questionColumns.length];
      for (int i = 0; i < header.length; i++) {
        for (int j = 0; j  < questionColumns.length; j++) {
          if (questionColumns[j].equals(header[i])) {
            questionIndices[j] = i;
          }
        }
      }

      //Array of indices of columns where the answers are located
      int[] answerIndices = new int[answerColumns.length];
      for (int i = 0; i < header.length; i++) {
        for (int j = 0; j  < answerColumns.length; j++) {
          if (answerColumns[j].equals(header[i])) {
            answerIndices[j] = i;
          }
        }
      }

      while ((line = br.readLine()) != null) {

        // use comma as separator
        String[] rawData = new String[0];
        rawData = splitCSVLine(line).toArray(rawData);
        for (int i = 0; i < questionIndices.length; i++) {
          if (!rawData[answerIndices[i]].equals("{}")) {
            sentences.add(new String[] {rawData[questionIndices[i]], rawData[answerIndices[i]], rawData[0]});
          }
        }

      }

    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      if (br != null) {
        try {
          br.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }

    return sentences;
  }

  private static void writeFile(List<String[]>sentences) {
    FileWriter fileWriter = null;
    PrintWriter printWriter = null;
    try {
      fileWriter = new FileWriter(outFile);
      printWriter = new PrintWriter(fileWriter);
      for (String[] sentenceArray : sentences) {
        for (String sentence : sentenceArray) {
          String tokens = StringUtils.join(new Sentence(sentence).words(), " ");
          printWriter.print(tokens + "\t");
        }
        printWriter.print("\n");
      }
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      if (printWriter != null) {
        printWriter.close();
      }
    }

  }

  public static void main(String[] args) {
    ArgumentParser.fillOptions(GenerateDataset.class, args);
    List<String[]> sentences = readSentences();
    writeFile(sentences);
  }
}
