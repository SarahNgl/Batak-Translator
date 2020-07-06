from math import log

kamus = open("/Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/BFS/kamus.txt").read().split() 
jumlahkata = dict((k, log((i+1)*log(len(kamus)))) for i,k in enumerate(kamus))
MaksimumKata = max(len(x) for x in kamus)

def cekkata(s):

    def best_match(i):
        PrediksiKata = enumerate(reversed(cost[max(0, i-MaksimumKata):i]))
        return min((c + jumlahkata.get(s[i-k-1:i], 9e999), k+1) for k,c in PrediksiKata)

    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))
#FIle yang di ambil itu dari sini. file translator nya
prediksi = open("/Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/Translator/output.txt",'r')

with open("/Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/BFS/outputBFS.txt", "w+") as f:
        print(cekkata(prediksi.read()), file=f)
        print(cekkata(prediksi.read())) 

