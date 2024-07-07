import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Read the text from your CV
text = """
Here's the plain text extracted from your CV:

---

**Jaynil Jaiswal**  
+1(984)325-5829  
San Diego  
jjaiswal@ucsd.edu  
GitHub: JaynilJaiswal  
LinkedIn: jaynil-jaiswal

---

### Education

**University of California, San Diego**   
Master of Science: Computer Science  

**Indian Institute of Technology, Roorkee**  
Bachelor of Technology: Computer Science & Engineering  

---

### Technical Strengths

**Languages**  
Python, Java, C, C++, Golang, TypeScript, Bash

**Big Data & Databases**  
Apache Hadoop, Spark, MySQL, Redis, AeroSpike, MongoDB

**Software Systems & ML**  
Tensorflow, PyTorch, DaskML, OpenCV, NLTK, OpenMP, CUDA, ARM SVE, Flask, ReactJS, Django, JUnit, RESTful APIs, SpringBoot

**Cloud & DevOps**  
Docker, Terraform, Kubernetes, AWS, Oracle Cloud Infrastructure, Confluence, JIRA

---

### Work Experience

**Member of Technical Staff | Oracle Corp., Bangalore, India** 
- Worked as main engineer for this key project to optimize the Dynamic Routing Gateway by making significant changes to an API for route filtering. This initiative resulted in a decrease of more than 10% in the total number of routes stored for large customers.
- Developed advanced DevOps tools and implemented Go routines to proactively monitor and manage the health of over 1200+ routers to streamline network operations.
- Explored in-memory databases (Redis, Aerospike) for developing a Bandwidth Broker serving 50000+ cloud customers.
- Played a key role in enhancing the Terraform provider plugin for Dynamic Routing Gateway, improving its functionality and .
- Collaboratively designed & developed DevOps tools with IC4 and IC5 team members.

**MLOps Co-op | Proton AutoML (Acquired by Cliently), Remote** 
- Architected and implemented a robust cloud infrastructure on AWS, capable of managing a significant data load of 160 files, each containing around 2-3GB of data.
- Engineered data pipelines that seamlessly integrated data from various sources such as Amazon RDS, Snowflake, Google Drive, and Dropbox. This ensured a consistent and reliable data flow, crucial for the  of the AutoML application.
- Boosted data processing efficiency by implementing data parallelization using DaskML. This innovative approach led to a notable speed increase of approximately 1.3x times, even for smaller workloads.
- Developed asynchronous APIs to handle a high volume of concurrent requests, with a maximum capacity of 5,000 requests for each server.

**Research Intern | Video Analytics Lab, IISc Bangalore** 
- Experimented with the fusion techniques of LDR images using Autoencoder architecture to de-ghost HDR images capturing motion. Achieved PSNR of 41db using BGU upsampling of LDR representation of the final merged image.
- Employed NVidia DGX-1 GPU cluster to conduct model training on 40,960 cores.

**Student Researcher | Machine Vision & Intelligence Lab, IIT Roorkee** 
- Surveyed the existing state-of-the-art text-to-speech evaluation techniques, identified research gaps and developed an objective metric to automatically evaluate the human-ness of machine-synthesized speech using Conditional Generative Adversarial Networks.
- Published my research as "A Generative Adversarial Network Based Ensemble Technique for Automatic Evaluation of Machine Synthesized Speech, ACPR 2019. Lecture Notes in Computer Science(), vol 12047. Springer"

---

### Publications

- "A Generative Adversarial Network based Ensemble Technique for Automatic Evaluation of Machine Synthesized Speech"  
  Jaynil Jaiswal, Ashutosh Chaubey, Bhimavarapu Sasi Kiran Reddy, Shashank Kashyap, Puneet Kumar, Balasubramanian Raman, Partha Pratim Roy  
  5th Asian Conference on Pattern Recognition, ACPR 2019, LNCS, Springer Nature, 2020

---

### Projects

**Optimized Matrix Multiplication on CUDA** 
- Implemented an efficient matrix multiplication algorithm using CUDA, leveraging shared memory for  optimization.
- Developed a 2D tiling approach to maximize data reuse and minimize global memory access latency, enhancing computational throughput.
- Achieved significant  gains, demonstrating high arithmetic intensity and efficient utilization of GPU resources.
- Validated the implementation using Nsight for  profiling, achieving a throughput of 3386 GFLOPS with minimal DRAM bandwidth usage.
- Demonstrated proficiency in CUDA programming, memory coalescing, and  optimization techniques for high- computing applications.

**A Simple Approach to Black-box Adversarial Attacks | University of California, San Diego** 
- Developed a novel adversarial model that uses a specialized loss function combining reconstruction similarity and adversarial impact to generate adversarial examples that maintain their label but mislead the neural network.
- Evaluated the model on MNIST and CIFAR-10 datasets, demonstrating significant impairment of classifier accuracy, with a drop in accuracy from 99% to 23% on MNIST and from 90% to 41% on CIFAR-10.
- Implemented a 3-layer autoencoder architecture for the generator and a pretrained 3-layer digit classifier for MNIST.
- Addressed the balance between generating effective adversarial examples and preserving their original label by fine-tuning the hyperparameter λ through empirical testing.
- Conducted experiments on textual data using a transformer encoder and LSTM generator, with mixed results due to computational limitations.

**Fleet Health Visualizer & Restarter | Oracle India Pvt. Ltd.** 
- We made a ops tool to monitor availability of the service by various health metrics at a glance for different sources of traffic.
- We can also visualize traffic distribution over different fleets to see if there is congestion on a host.
- We can promptly respond to any incident and restore the availability by doing bulk restarts of the hosts.

**Route Filtering | Oracle India Pvt. Ltd.** 
- Created a flag to enable aggregating of dynamic routes rules using CIDR for virtual cloud network.
- This reduces the amount of route rules needed to store in our route table and enhance .

**OLIVIA (Optimized Lightweight Intelligent Voice Interactive Assistant)** 
- Developed a versatile voice assistant that can be tailored to different industries and carry out multiple tasks simultaneously.
- Implemented primary modules such as Speech-to-text, Text-to-speech, Natural-Language-Understanding (NLU) shell, Chatbot, Question And Answering, and Deep Punctuation.
- Integrated secondary modules like Snap/Clap activation, AdultFilter, Voice Authentication, and TextSummarization to enhance the assistant's  and user experience.
- The feature pool included functionalities such as Time, Weather, Schedule List/Timetable, Play Music, Find Information, Send Message, Send Email, Play best videos related to a query, Handle Call and extract information or send a relevant information, and Trending news.

**Few-Shot Unsupervised Image-to-Image Translation** 
- Applied Generative Adversarial Networks (GANs) for unsupervised image translation, innovatively extending the state-of-the-art model to enhance generalization across classes.
- Employed attention mechanisms, leveraging multiple sample images from various classes, to proficiently learn and adapt to new classes.
- Validated  through InceptionNet V3 scores pretrained on the target class, achieving a notable 0.73 accuracy in Top-5 test solely with Vanilla-GAN.

**Cross Domain Sentiment Analysis | IIT Roorkee**
- The approach implements this idea in the context of neural network architectures that are trained on labeled data from the source domain and unlabeled data from the target domain (no labeled target-domain data is necessary).
- We show that this adaptation behavior can be achieved in almost any feed-forward model by augmenting it with a few standard layers and a new gradient reversal layer.

**Data augmenting using Variational Auto-Encoder**
- Implemented Variational Auto-Encoder using Tensorflow framework. Used MNIST dataset to train the network to learn representations of different numbers in many handwritings.
- I used the trained encoder to interpolate features of existing images to decode new images.

**AEREM | Microsoft Code.Fun.Do 2017**
- An application written in C++ which make use of OpenCV to track motion of a pointer in front a camera and recognize the character written in air in front of it using OCR. The prototype contained capability to record letters written in air in front of a camera.

---

### Leadership & Extracurriculars

**Senior Executive | Student Technical Council, IIT Roorkee**
- Led a team of 20 students in organizing technical events, workshops, and hackathons.
- Successfully executed multiple high-profile events, enhancing technical engagement on campus.

**Volunteer | Teach for India** 
- Taught underprivileged children in remote areas, focusing on basic education and computer literacy.

---

This text can now be used to generate the word cloud.
"""  # Add your entire CV text here

# Load the mask image
mask = np.array(Image.open("images/man_shape.png"))

# Define stopwords
stopwords = set(STOPWORDS)
stopwords.update(["will", "using", "work", "experience", "developed", "project", "implemented", "used"])

# Create the word cloud
wordcloud = WordCloud(
    background_color="white",
    mask=mask,
    stopwords=stopwords,
    contour_width=3,
    contour_color='black'
).generate(text)

# plt.imshow(mask)

# Display the word cloud
plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
