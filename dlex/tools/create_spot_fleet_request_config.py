import sys
import json
import base64

with open('bin/launch.sh') as f:
    user_data = base64.b64encode(f.read())

config = {
  "TargetCapacity": 1,
  "AllocationStrategy": "diversified",
  "IamFleetRole": "arn:aws:iam::127193726529:role/dlex",
  "LaunchSpecifications": [
    {
      "ImageId": "ami-0f9e8c4a1305ecd22",
      "KeyName": "dlex",
      "SecurityGroups": [
        {
          "GroupId": "sg-0889317e67faef5b0"
        }
      ],
      "InstanceType": "p2.xlarge",
      "UserData": user_data,
      "IamInstanceProfile": {
        "Arn": "arn:aws:iam::127193726529:instance-profile/dlex"
      }
    }
  ]
}

with open(sys.argv[1], 'w') as f:
    f.write(json.dumps(config))
