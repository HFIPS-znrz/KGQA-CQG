import codecs

def delete_same(str_, pad):
    new_str = []
    str_words = str_.split(" ")
    i = 0
    if i + 2*pad  > len(str_words):
        return str_
    while i + 2*pad <= len(str_words):
        current_words = str_words[i:i+pad]
        i= i+pad
        while i + pad <= len(str_words) and " ".join(current_words) == " ".join(str_words[i:i+pad]):

            i = i+pad
        new_str+=current_words

    if i<len(str_words):
        new_str += str_words[i:]
    return " ".join(new_str)


def read_entity(path):
    sr = codecs.open(path, "r", "utf-8")
    lines = sr.readlines()
    entity_name = []
    for line in lines:
        line = line.strip()

        line = delete_same(line,1)
        line = delete_same(line,2)
        line = line.replace('"',"")
        entity_name.append(line)
    return entity_name



def read_template(path, entity1, entity2):
    sr = codecs.open(path, "r", "utf-8")
    sw = codecs.open("../../../code/generation/OpenNMT/final_output_single.txt", 'w', 'utf-8')
    lines = sr.readlines()
    final_answer = []
    for i,line in enumerate(lines):
        line = line.strip()
        line = line.split(" <TSP> ")[0]

        line = delete_same(line,1)
        line = delete_same(line,2)
        line = line.replace('"',"")
        # print('line', line)
        if "<e>" in line:
            postion1 = line.index("<e>")
            if postion1+len("<e>") >= len(line):
                print(line)
            

            first = line[0:postion1+len("<e>")].replace("<e>",entity1[i])
            
            second = line[postion1+len("<e>"):].replace("<e>", entity2[i])
            final_answer.append(first + second)
            sw.write(first+second+"\n")
        else:
            sw.write(line+"\n")
    sw.close()


def mer():
    entity1 = read_entity("../../../code/generation/OpenNMT/pred_entity1.txt")
    entity2 = read_entity("../../../code/generation/OpenNMT/pred_entity2.txt")
    read_template("../../../code/generation/OpenNMT/output.txt", entity1, entity2)
    #read_template("multi-without-copy.txt", entity1, entity2)
    with open('../../../code/generation/OpenNMT/final_output_single.txt', 'r', encoding='utf-8') as f:
        first_line = f.readline().strip().replace(" ", "").replace("\n", "")
        return first_line

if __name__ == '__main__':
    clari = mer()
    print(clari)