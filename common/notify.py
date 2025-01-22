import requests
import sys

if __name__ == "__main__":
    body = sys.argv[1]
    r = requests.post("https://ntfy.sh/XPCI-training", data=f"Done: {body}".encode(encoding='utf-8'))
    print(r.status_code, r.reason)