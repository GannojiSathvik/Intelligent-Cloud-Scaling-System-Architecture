"""
AWS Services Verification Script
Checks all components of the Intelligent Cloud Scaling System
"""
import boto3
from datetime import datetime
import json

# Configuration
S3_BUCKET_NAME = "my-intelligent-scaling-data-bucket"
ASG_NAME = "intelligent-scaling-demo-sathvik"
LAMBDA_FUNCTIONS = ["DataCollectorFunction", "ScalingLogicFunction"]

def print_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_s3_bucket():
    """Check S3 bucket status"""
    print_header("📦 S3 BUCKET STATUS")
    s3 = boto3.client('s3')
    
    try:
        # Check if bucket exists
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"✅ Bucket exists: {S3_BUCKET_NAME}")
        
        # List objects
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=10)
        if 'Contents' in response:
            print(f"✅ Files in bucket: {len(response['Contents'])}")
            for obj in response['Contents'][:5]:
                size_kb = obj['Size'] / 1024
                print(f"   • {obj['Key']} ({size_kb:.2f} KB) - {obj['LastModified']}")
        else:
            print("⚠️  Bucket is empty")
        
        # Check for data file
        try:
            obj = s3.head_object(Bucket=S3_BUCKET_NAME, Key='multi_metric_data.csv')
            size_mb = obj['ContentLength'] / (1024 * 1024)
            print(f"✅ Data file exists: multi_metric_data.csv ({size_mb:.2f} MB)")
            print(f"   Last modified: {obj['LastModified']}")
        except:
            print("❌ Data file not found: multi_metric_data.csv")
        
        return True
    except Exception as e:
        print(f"❌ Bucket check failed: {e}")
        print(f"\n💡 To create bucket, run:")
        print(f"   aws s3 mb s3://{S3_BUCKET_NAME}")
        return False

def check_lambda_functions():
    """Check Lambda functions status"""
    print_header("⚡ LAMBDA FUNCTIONS STATUS")
    lambda_client = boto3.client('lambda')
    
    all_ok = True
    for func_name in LAMBDA_FUNCTIONS:
        try:
            response = lambda_client.get_function(FunctionName=func_name)
            config = response['Configuration']
            
            print(f"\n✅ {func_name}")
            print(f"   • Runtime: {config['Runtime']}")
            print(f"   • Memory: {config['MemorySize']} MB")
            print(f"   • Timeout: {config['Timeout']}s")
            print(f"   • Last Modified: {config['LastModified']}")
            
            # Check recent invocations
            logs_client = boto3.client('logs')
            log_group = f"/aws/lambda/{func_name}"
            try:
                streams = logs_client.describe_log_streams(
                    logGroupName=log_group,
                    orderBy='LastEventTime',
                    descending=True,
                    limit=1
                )
                if streams['logStreams']:
                    last_event = streams['logStreams'][0].get('lastEventTimestamp', 0)
                    last_time = datetime.fromtimestamp(last_event/1000)
                    print(f"   • Last execution: {last_time}")
            except:
                print(f"   ⚠️  No recent executions found")
                
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"\n❌ {func_name} - NOT DEPLOYED")
            all_ok = False
        except Exception as e:
            print(f"\n❌ {func_name} - Error: {e}")
            all_ok = False
    
    return all_ok

def check_auto_scaling_group():
    """Check Auto Scaling Group status"""
    print_header("🔄 AUTO SCALING GROUP STATUS")
    asg_client = boto3.client('autoscaling')
    
    try:
        response = asg_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[ASG_NAME]
        )
        
        if not response['AutoScalingGroups']:
            print(f"❌ ASG not found: {ASG_NAME}")
            return False
        
        asg = response['AutoScalingGroups'][0]
        print(f"✅ ASG exists: {ASG_NAME}")
        print(f"   • Desired Capacity: {asg['DesiredCapacity']}")
        print(f"   • Min Size: {asg['MinSize']}")
        print(f"   • Max Size: {asg['MaxSize']}")
        print(f"   • Current Instances: {len(asg['Instances'])}")
        
        # List instances
        if asg['Instances']:
            print(f"\n   Instances:")
            for inst in asg['Instances']:
                print(f"   • {inst['InstanceId']} - {inst['HealthStatus']} - {inst['LifecycleState']}")
        
        # Check recent scaling activities
        activities = asg_client.describe_scaling_activities(
            AutoScalingGroupName=ASG_NAME,
            MaxRecords=5
        )
        
        if activities['Activities']:
            print(f"\n   Recent Scaling Activities:")
            for activity in activities['Activities'][:3]:
                print(f"   • {activity['StartTime']}: {activity['Description']}")
                print(f"     Status: {activity['StatusCode']}")
        
        return True
    except Exception as e:
        print(f"❌ ASG check failed: {e}")
        return False

def check_eventbridge_rules():
    """Check EventBridge rules for Lambda triggers"""
    print_header("⏰ EVENTBRIDGE RULES STATUS")
    events_client = boto3.client('events')
    
    try:
        response = events_client.list_rules(NamePrefix='ScalingLogic')
        
        if response['Rules']:
            for rule in response['Rules']:
                print(f"\n✅ Rule: {rule['Name']}")
                print(f"   • Schedule: {rule.get('ScheduleExpression', 'N/A')}")
                print(f"   • State: {rule['State']}")
                
                # Get targets
                targets = events_client.list_targets_by_rule(Rule=rule['Name'])
                print(f"   • Targets: {len(targets['Targets'])}")
        else:
            print("⚠️  No EventBridge rules found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ EventBridge check failed: {e}")
        return False

def check_cloudwatch_metrics():
    """Check CloudWatch metrics"""
    print_header("📊 CLOUDWATCH METRICS STATUS")
    cw_client = boto3.client('cloudwatch')
    
    try:
        # Check for ASG metrics
        response = cw_client.list_metrics(
            Namespace='AWS/EC2',
            Dimensions=[{'Name': 'AutoScalingGroupName', 'Value': ASG_NAME}]
        )
        
        if response['Metrics']:
            print(f"✅ CloudWatch metrics available for ASG")
            metric_names = set([m['MetricName'] for m in response['Metrics']])
            print(f"   Available metrics: {', '.join(list(metric_names)[:5])}")
        else:
            print("⚠️  No metrics found for ASG")
        
        return True
    except Exception as e:
        print(f"❌ CloudWatch check failed: {e}")
        return False

def generate_report():
    """Generate comprehensive status report"""
    print_header("📋 COMPREHENSIVE STATUS REPORT")
    
    results = {
        'S3 Bucket': check_s3_bucket(),
        'Lambda Functions': check_lambda_functions(),
        'Auto Scaling Group': check_auto_scaling_group(),
        'EventBridge Rules': check_eventbridge_rules(),
        'CloudWatch Metrics': check_cloudwatch_metrics()
    }
    
    print_header("🎯 SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nSystem Status: {passed}/{total} components operational\n")
    
    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {component}")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL - READY FOR DEMONSTRATION!")
    elif passed >= total * 0.7:
        print("\n⚠️  PARTIAL DEPLOYMENT - Some components need attention")
    else:
        print("\n❌ SYSTEM NOT READY - Multiple components need deployment")
    
    # Save report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'results': {k: ('PASS' if v else 'FAIL') for k, v in results.items()},
        'score': f"{passed}/{total}"
    }
    
    with open('aws_verification_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Report saved: aws_verification_report.json")

if __name__ == "__main__":
    print("="*80)
    print("  🔍 AWS INTELLIGENT CLOUD SCALING SYSTEM - VERIFICATION")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    
    try:
        generate_report()
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        print("\n💡 Make sure AWS CLI is configured:")
        print("   aws configure")
