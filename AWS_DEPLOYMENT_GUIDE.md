# AWS App Runner Deployment Guide

This guide walks you through deploying the Retail Autonomous Researcher on AWS App Runner using the AWS Console.

## Prerequisites

1. AWS Account (create one if needed at https://aws.amazon.com)
2. Your Docker Hub image: `akshatdubey21/retail-autonomous-researcher:latest`
3. API Keys:
   - GROQ_API_KEY: your Groq API key
   - TAVILY_API_KEY: your Tavily API key

## Step-by-Step Deployment

### 1. Open AWS Console
- Go to https://console.aws.amazon.com
- Sign in with your AWS account
- Make sure region is set to **us-east-1** (top right)

### 2. Navigate to App Runner
- Search for "App Runner" in the search bar
- Click "App Runner"
- Click "Create an App Runner service"

### 3. Configure Source
- **Source**: Select "Container registry"
- **Provider**: Select "Docker Hub"
- **Image URI**: `akshatdubey21/retail-autonomous-researcher:latest`
- **Registry access role**: Leave as "Create new service role"
- Click **Next**

### 4. Configure Service
- **Service name**: `retail-autonomous-researcher`
- **Port**: `8501`
- **CPU**: `1 vCPU` (or `2 vCPU` if budget allows)
- **Memory**: `2 GB` (or `4 GB` for better performance)
- **Environment variables**: Add the following (click "Add environment variable"):

Important: enter these as separate key/value pairs in the Elastic Beanstalk environment settings. Do not paste them as a single plain-text block.

| Key | Value |
|-----|-------|
| GROQ_API_KEY | your Groq API key |
| TAVILY_API_KEY | your Tavily API key |
| GROQ_MODEL | llama-3.3-70b-versatile |

- Click **Next**

### 5. Configure Deployment
- **Deployment trigger**: Select "Manual" (deploy when you push; select "Automatic" if you want auto-redeploy on Docker Hub image update)
- Click **Next**

### 6. Review and Deploy
- Review all settings
- Click **Create & Deploy**

### 7. Wait for Deployment (5-10 minutes)
- AWS will:
  - Pull your Docker image from Docker Hub
  - Build/start the container
  - Allocate a public URL
- Once status changes to **Running** (green), you're done!

### 8. Access Your App
- Click the **Default domain** link
- Your Streamlit app will open at `https://<service-name>-*.awsapprunner.com`
- Try submitting a retail research query!

## Cost Estimates (us-east-1)

- **1 vCPU + 2 GB RAM**: ~$5-7/month (if running 24/7)
- **2 vCPU + 4 GB RAM**: ~$10-14/month (if running 24/7)
- First-time free tier: $1 free credit for new AWS accounts

## Troubleshooting

### Service won't start?
- Check "Recent deployments" tab for logs
- Verify environment variables are set correctly
- Check if Groq/Tavily API keys are valid

### `tavily.errors.InvalidAPIKeyError`
- Confirm `TAVILY_API_KEY` exists in the environment properties for the EB environment.
- Make sure the value is copied exactly and does not include quotes, spaces, or line breaks.
- If you pasted a plain-text block into the environment field, replace it with separate key/value entries for each variable.

### Slow response times?
- Increase CPU/Memory (1→2 vCPU, 2→4 GB)
- This will cost more but improve performance

### Reports not saving?
- Reports are saved locally in container (ephemeral)
- They disappear after restart
- For persistent storage, use S3 (requires code changes)

## Security Notes

**IMPORTANT: Before going to production:**
1. Your API keys are visible in environment variables
2. Consider using **AWS Secrets Manager** instead:
   - App Runner → Configuration → Secrets
   - Store keys securely
3. Rotate API keys periodically

## Next Steps

1. Deploy using steps above
2. Test with a sample query
3. Share the public URL with stakeholders
4. Monitor costs in AWS Cost Explorer

## Support

- AWS App Runner docs: https://docs.aws.amazon.com/apprunner/
- Troubleshoot: Check service logs in App Runner console
