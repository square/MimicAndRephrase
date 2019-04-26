package ai.eloquent.rephrasing;

import edu.stanford.nlp.simple.Sentence;

import java.util.*;

/**
 * Rule based Emotive rephraser
 *
 * @author <a href="mailto:angel@eloquent.ai">*Angel Chang</a>
 */
public class RuleBasedEmotiveRephraser extends RuleBasedRephraser {

    public enum Sentiment { Positive, Negative }
    static final Map<Sentiment, List<String>> EmotivePhrases = new HashMap<>();
    static final List<String> endPhrases = Arrays.asList(
            "that happened",
            "about that"
    );

    public static Map<Sentiment, List<String>> getEmotivePhrases() {
        if (EmotivePhrases.size() == 0) {
            EmotivePhrases.put(Sentiment.Positive, Arrays.asList(
                    "Good to hear",
                    "I am glad",
                    "I am happy",
                    "I'm glad",
                    "I'm happy"
            ));
            EmotivePhrases.put(Sentiment.Negative, Arrays.asList(
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

    public String getStartPhrase(Sentiment sentiment) {
        List<String> startPhrases = getEmotivePhrases().get(sentiment);
        int i = random.nextInt(startPhrases.size());
        return startPhrases.get(i);
    }

    public String getEndPhrase() {
        int i = random.nextInt(endPhrases.size());
        return endPhrases.get(i);
    }

    public Optional<String> rephrased(Sentence sentence, Sentiment sentiment, boolean condensed) {
        //Get initial phrase based on sentiment
        String start = getStartPhrase(sentiment);

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

    public String rephrasedGeneric(Sentiment sentiment) {
        //Get initial phrase based on sentiment
        String start = getStartPhrase(sentiment);
        String end = getEndPhrase();
        return start + " " + end;
    }

    protected Sentence replacePronouns(Sentence sentence) {
        return replacePronouns(sentence, 0, true);
    }
}
