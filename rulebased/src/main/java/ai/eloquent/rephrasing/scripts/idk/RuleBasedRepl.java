package ai.eloquent.rephrasing.scripts.idk;

import ai.eloquent.rephrasing.RuleBasedIDKRephraser;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.util.ArgumentParser;

import java.io.IOException;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.Scanner;

/**
 * Repl to manually test out rule based IDK and emotive rephrasers
 *
 * @author <a href="mailto:*angel@eloquent.ai">*Angel Chang</a>
 */
public class RuleBasedRepl {

    public static void repl() {
        // Test a sentence.
        Scanner in = new Scanner(System.in);
        System.out.println("Enter a request");
        RuleBasedIDKRephraser rephraser = new RuleBasedIDKRephraser();
        while (true) {
            System.out.print("> ");
            String prompt;
            try {
                prompt = in.nextLine().trim();
                if (prompt.length() == 0) continue;
                Optional<String> rephrased = rephraser.rephrased(new Sentence(prompt));
                if (rephrased.isPresent()) {
                    System.out.println(rephrased);
                } else {
                    System.out.println("I don't know how to handle '" + prompt + "'");
                }
            } catch (NoSuchElementException ignored) {
                break;
            }
        }
    }

    public static void main(String[] args) throws IOException {
        repl();
    }
}
