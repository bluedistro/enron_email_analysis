import logging.handlers
log = logging.getLogger()
fh = logging.handlers.RotatingFileHandler("input/emails.csv", maxBytes=2**20*100, backupCount=100) 
# 100 MB each, up to a maximum of 100 files
log.addHandler(fh)
log.setLevel(logging.INFO)
f = open("input/chunked_mails.csv")
while True:
    log.info(f.readline().strip())