package ai.eloquent.rephrasing;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.ArraySet;
import edu.stanford.nlp.util.IntPair;

import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Shared based class for rule based rephrasing
 *
 * @author <a href="mailto:angel@eloquent.ai">*Angel Chang</a>
 */
public class RuleBasedRephraser {
    // Patterns of first person pronouns and second person replacements
    private static String[] firstPerson = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"};
    private static String[] firstPersonReplace = {"you", "you", "your", "yours", "yourself", "you", "you", "your", "yours"};

    // Patterns of second person pronouns and first person replacements
    // (you is not included since we need to use the dependency graph to determine if it should be me or I)
    private static String[] secondPerson = {"your", "yours", "yourself"};
    private static String[] secondPersonReplace = {"my", "mine", "myself"};

    private static String[] firstPersonVerb = {"am", "'m", "was"};
    private static String[] secondPersonVerb = {"are", "are", "were"};

    protected Sentence replaceWord(Sentence sentence, int index, String newWord) {
        ArrayList<String> newWords = new ArrayList<>(sentence.words());
        newWords.remove(index);
        newWords.add(index, newWord);
        return new Sentence(newWords);
    }

    protected Sentence matchAndReplaceWord(Sentence sentence, int index, String[] matchWords, String[] replacements) {
        assert(matchWords.length == replacements.length);
        String word = sentence.word(index);
        for (int j = 0; j < matchWords.length; j++) {
            if (word.equalsIgnoreCase(matchWords[j])) {
                return replaceWord(sentence, index, replacements[j]);
            }
        }
        return sentence;
    }

    protected Sentence replacePronouns(Sentence sentence, int start, boolean replaceNextVerb) {
        Sentence workingSentence = sentence;
        int sentenceLength = workingSentence.words().size();
        for (int i = start; i < sentenceLength; i++) {
            String word = workingSentence.word(i);
            for (int j = 0; j < firstPerson.length; j++) {
                if (word.equalsIgnoreCase(firstPerson[j])) {
                    workingSentence = replaceWord(workingSentence, i, firstPersonReplace[j]);
                    if (replaceNextVerb && i+1 < sentenceLength) {
                        workingSentence = matchAndReplaceWord(workingSentence, i+1, firstPersonVerb, secondPersonVerb);
                    }
                    break;
                }
            }

            for (int j = 0; j < secondPerson.length; j++) {
                if (word.equalsIgnoreCase(secondPerson[j])) {
                    workingSentence = replaceWord(workingSentence, i, secondPersonReplace[j]);
                    if (replaceNextVerb && i+1 < sentenceLength) {
                        workingSentence = matchAndReplaceWord(workingSentence, i+1, secondPersonVerb, firstPersonVerb);
                    }
                    break;
                }
            }

            if (word.equalsIgnoreCase("You")) {
                Optional<String> dependencies = workingSentence.incomingDependencyLabel(i);

                if (dependencies.isPresent() && dependencies.get().equalsIgnoreCase("nsubj")) {
                    workingSentence = replaceWord(workingSentence, i, "I");
                    if (replaceNextVerb && i+1 < sentenceLength) {
                        workingSentence = matchAndReplaceWord(workingSentence, i+1, secondPersonVerb, firstPersonVerb);
                    }
                } else {
                    workingSentence = replaceWord(workingSentence, i, "me");
                }
            }
        }
        return workingSentence;
    }

    // Adjust captalization so the first word is not capitalized (unless it's needed)
    protected Sentence adjustCapitalization(Sentence sentence) {
        if (sentence.words().size() > 0) {
            String word = sentence.word(0);
            boolean needCapitalize = "NNP".equals(sentence.posTag(0)) || "I".equals(word);
            if (!needCapitalize) {
                if (Character.isUpperCase(word.charAt(0))) {
                    sentence = replaceWord(sentence, 0, word.toLowerCase());
                }
            }
        }
        return sentence;
    }



    protected Tree dfs(Tree nodeX, Predicate<Tree> isMatch, Predicate<Tree> isExpandable) {
        Stack<Tree> toVisit = new Stack<>();
        toVisit.add(nodeX);
        while (!toVisit.isEmpty()) {
            Tree current = toVisit.pop();
            if (isMatch.test(current)) {
                return current;
            }
            for (int iChild = 0; iChild < current.numChildren(); iChild++) {
                Tree child = current.getChild(current.numChildren() - iChild - 1);
                if (isExpandable.test(child)) {
                    toVisit.push(child);
                }
            }
        }
        return null;
    }

    protected Sentence shortenUsingConstituencyParse(Sentence sentence) {
        Set<String> slabels = new ArraySet<>();
        slabels.add("ROOT");
        slabels.add("S");
        Tree t = sentence.parse();
        Tree sc = dfs(t, x -> !slabels.contains(x.value()), x -> true);
        if (sc != null) {
            Tree p = sc.parent(t);
            List<Word> words = p.yieldWords();
            return new Sentence(words.stream().map( x -> x.word()).collect(Collectors.toList()));
        }
        return sentence;
    }

    protected Sentence shorten(Sentence sentence) {
        return shortenUsingConstituencyParse(sentence);
    }

}
