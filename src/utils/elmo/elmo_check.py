from elmo import Elmo, batch_to_ids

elmo = Elmo.get_default(3)
elmo.cuda()

batch = [["hi", "there"], ["what", "is", "up"]]

char_ids = batch_to_ids(batch).cuda()
embeddings = elmo(char_ids)
print(embeddings["elmo_representations"])
