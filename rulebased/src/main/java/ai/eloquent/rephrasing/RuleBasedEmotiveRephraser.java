package ai.eloquent.rephrasing;

import edu.stanford.nlp.simple.Sentence;

import java.util.*;

/**
 * Rule based Emotive rephraser
 *
 * @author <a href="mailto:angel@eloquent.ai">*Angel Chang</a>
 */
public class RuleBasedEmotiveRephraser extends RuleBasedRephraser {

    public enum Emotive { Positive, Negative }
    static final Map<Emotive, List<String>> EmotivePhrases = new HashMap<>();
    static final List<String> endPhrases = Arrays.asList(
            "that happened",
            "about that"
    );

    public static Map<Emotive, List<String>> getEmotivePhrases() {
        if (EmotivePhrases.size() == 0) {
            EmotivePhrases.put(Emotive.Positive, Arrays.asList(
                    "Good to hear",
                    "I am glad",
                    "I am happy",
                    "I'm glad",
                    "I'm happy"
            ));
            EmotivePhrases.put(Emotive.Negative, Arrays.asList(
                    "I am sorry",
                    "Sorry to hear",
                    "Sorry",
                    "I am sad",
                    "I'm sorry"
            ));
        }
        return EmotivePhrases;
    }

    Random random = new Random();

    public void setSeed(long seed) {
        random.setSeed(seed);
    }

    public String getStartPhrase(Emotive Emotive) {
        List<String> startPhrases = getEmotivePhrases().get(Emotive);
        int i = random.nextInt(startPhrases.size());
        return startPhrases.get(i);
    }

    public String getEndPhrase() {
        int i = random.nextInt(endPhrases.size());
        return endPhrases.get(i);
    }

    public Optional<String> rephrased(Sentence sentence, Emotive Emotive, boolean condensed) {
        //Get initial phrase based on Emotive
        String start = getStartPhrase(Emotive);

        //Preprocess the sentence by switching the point of view of pronouns i.e. I to you
        Sentence rephrased = replacePronouns(sentence);

        //Make sure not upper cased
        rephrased = adjustCapitalization(rephrased);

        if (condensed) {
            rephrased = shorten(rephrased);
        }


        //Let's ignore the condensed flag
        return Optional.of(start + " " + rephrased);
    }

    public String rephrasedGeneric(Emotive Emotive) {
        //Get initial phrase based on Emotive
        String start = getStartPhrase(Emotive);
        String end = getEndPhrase();
        return start + " " + end;
    }

    protected Sentence replacePronouns(Sentence sentence) {
        return replacePronouns(sentence, 0, true);
    }
}
