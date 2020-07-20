# Pythono3 code to rename multiple 
# files in a directory or folder 

# importing os module 
import os 

# Function to rename multiple files 
def main(): 

	for count, filename in enumerate(os.listdir("/Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/Translator/outs/Foto")): 
		dst ="XX" + str(count) + ".jpg"
		src ='/Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/Translator/outs/Foto/'+ filename 
		dst ='/Users/tania/Desktop/Desktop/ORIGINAL/Code/Tugas_Akhir/AksaraBatak/Translator/outs/JPEGImages/'+ dst 
		
		# rename() function will 
		# rename all the files 
		os.rename(src, dst) 

# Driver Code 
if __name__ == '__main__': 
	
	# Calling main() function 
	main() 
