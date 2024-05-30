This script is programmed to create websites with AI. I have added about 225 templates to a dataset in Vertex AI and here is the process:

1. OpenCV/LayoutLM will select a template file based on the user data input

2. YOLO will detect the images in the website template file

3. Pexels API will change the detected images to images related to the user data input (Industry)

4. T5 will then change the written content of the website relative to user input data and add 'Made with AI by Carrot' at the footer

5. Website will be deployed to a subdomain

That's all for AI tool

Once its deployed, we can connect the reseller API with the AI tool so that it grabs a subdomain/domain if they purchase
12:06 AM
Pexels API key to add to code: Crz09TTIZYRABYgtgn1I7fLeanJk2ti0dzuIpRtY9lgzuspjHmzPQ7mv
12:06 AM
Please also change 'analyze_layoutlm():' to 'analyze_opencv():' as we will be using OpenCV instead of LayoutLM
12:07 AM
Then debug and deploy
12:07 AM
