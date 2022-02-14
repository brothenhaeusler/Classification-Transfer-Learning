# google images downloader try á la
# #https://github.com/hardikvasa/google-images-download/issues/316#issuecomment-698362732
# doesn't work

# what works:
# https://github.com/Joeclinton1/google-images-download

#$ cd /Users/br/Downloads/google-images-download-patch-1/
#$ python3 setup.py install
#$ python3 

from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()
arguments = {"keywords":"Trump","limit":50,"print_urls":True}
paths = response.download(arguments)

from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()
arguments = {"keywords":"Barack Obama ","limit":100,"print_urls":True, "format":"jpg", "aspect_ratio":"square", "size":">400*300"}

arguments = {"keywords":"Joe Biden Closeup","limit":100,"print_urls":True, "format":"jpg", "aspect_ratio":"square", "size":">400*300"}
#arguments = {"keywords":"Donald Trump Closeup","limit":200,"print_urls":True, "format":"jpg", "aspect_ratio":"square", "size":">400*300","offset":100}
#arguments = {"keywords":"Donald Trump Closeup","limit":200,"print_urls":True, "format":"jpg", "aspect_ratio":"square"}


#arguments = {"keywords":"Donald Trump Closeup","limit":1000,"print_urls":True, "format":"jpg", "aspect_ratio":"square", "size":">400*300", "delay":1}

#arguments = {"keywords":"Barack Obama Closeup","limit":100,"print_urls":True, "format":"jpg", "aspect_ratio":"square", "size":">400*300"}

paths = response.download(arguments)




# setting offset doesn't work (you can't specify less pictures)
# setting a delay between 2 pictures doesn't work
# so probably lots of work to be done here. 
# Bill Clinton Closeup is really bad
