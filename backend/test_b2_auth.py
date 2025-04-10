from b2sdk.v2 import InMemoryAccountInfo, B2Api

print("Testing Backblaze B2 authorization...")
info = InMemoryAccountInfo()
b2_api = B2Api(info)
B2_KEY_ID = "005f0b86c723e100000000004"
B2_APPLICATION_KEY = "K0054i9WdpAQU2QKKA9dn8VHsasOJss"

try:
    print(f"Authorizing with Key ID: {B2_KEY_ID}")
    print(f"Application Key: {B2_APPLICATION_KEY}")
    b2_api.authorize_account("production", B2_APPLICATION_KEY, B2_KEY_ID)
    print("Authorization successful!")
    bucket = b2_api.get_bucket_by_name('ai-fitness-videos')
    print(f"Bucket found: {bucket.bucket_name}")
except Exception as e:
    print(f"Error: {e}")