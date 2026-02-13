"use client";

import { useState } from 'react';
import { useTheme } from 'next-themes';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faMoon, faSun, faBars, faRobot, faCode, faMicrochip, 
  faChartLine, faServer, faNetworkWired, faStream, 
  faBuilding, faExternalLinkAlt, faFilePdf, faEnvelope, 
  faMapMarkerAlt, faBookOpen, faArrowRight, faCloud 
} from '@fortawesome/free-solid-svg-icons';
import { 
  faGithub, faLinkedinIn, faMediumM, faTwitter, faHuggingFace 
} from '@fortawesome/free-brands-svg-icons';

// --- DATA SECTION: EASILY EDIT YOUR INFO HERE ---

const projects = [
  {
    title: "LLM-Powered Search Engine",
    desc: "A custom search engine powered by large language models that allows natural language queries across my portfolio projects and experience.",
    tags: ["Python", "HuggingFace", "FastAPI"],
    link: "https://huggingface.co/spaces/JaynilJaiswal/portfolio",
    icon: faRobot,
    color: "from-blue-500 to-blue-700"
  },
  {
    title: "Durable Multi-Agent Orchestration",
    desc: "Architected a fault-tolerant agent orchestration platform using Temporal.io, enabling durable workflows, stateful orchestration, and 'Skeletonizer' memory management.",
    tags: ["Temporal.io", "Java", "SpringBoot"],
    link: "#",
    icon: faCode,
    color: "from-purple-500 to-pink-500"
  },
  {
    title: "Matrix Multiplication on GeneSys",
    desc: "Programmed GeneSys hardware with a 4x4 systolic array of PEs for efficient matrix multiplication, leveraging iterator tables and instruction scheduling.",
    tags: ["Hardware Acceleration", "ISA"],
    link: "https://github.com/JaynilJaiswal/GeneSys-Matrix-Multiplication",
    icon: faMicrochip,
    color: "from-green-500 to-teal-500"
  },
  {
    title: "Sales Data Warehouse & BI",
    desc: "Designed a scalable OLAP Data Warehouse in PostgreSQL. Built ETL pipelines handling 10M+ daily records for a Looker Studio dashboard.",
    tags: ["PostgreSQL", "Python ETL", "Looker"],
    link: "https://github.com/JaynilJaiswal",
    icon: faChartLine,
    color: "from-blue-400 to-cyan-500"
  },
  {
    title: "Enterprise AI Gateway",
    desc: "Built a high-performance AI Gateway using KServe for model serving and Redis for semantic caching. Reduced query latency by 60%.",
    tags: ["KServe", "Redis", "LangChain"],
    link: "https://github.com/JaynilJaiswal",
    icon: faServer,
    color: "from-purple-500 to-pink-500"
  },
  {
    title: "Optimized Matrix Multiplication",
    desc: "Engineered a highly optimized SGEMM kernel in CUDA, achieving 3386 GFLOPS. Utilized 2D shared memory tiling and coalesced memory reads.",
    tags: ["CUDA", "C++", "Nsight"],
    link: "https://github.com/cse260-sp24/pa2-pa2-jjaiswal-ang057",
    icon: faMicrochip,
    color: "from-green-600 to-teal-600"
  },
  {
    title: "Distributed Cache with gRPC",
    desc: "Built a high-availability distributed in-memory cache in Go. Implemented custom Gossip protocol and gRPC bidirectional streaming.",
    tags: ["Go", "gRPC", "Consistent Hashing"],
    link: "https://github.com/JaynilJaiswal",
    icon: faNetworkWired,
    color: "from-cyan-600 to-blue-700"
  },
  {
    title: "Instruction-Following LLM Fine-Tuning",
    desc: "Fine-tuned FLAN-T5-XL and Mistral-7B on Bitext dataset using LoRA, achieving 28% ROUGE-L improvement with 60% memory reduction.",
    tags: ["LoRA", "FLAN-T5", "Mistral-7B"],
    link: "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    icon: faRobot,
    color: "from-yellow-500 to-orange-500"
  },
  {
    title: "Streaming ETL Pipeline",
    desc: "Automated a scalable real-time ETL pipeline with Apache Kafka, MySQL, and Airflow for toll traffic analysis, handling 1M+ records.",
    tags: ["Kafka", "Airflow", "MySQL"],
    link: "https://github.com/JaynilJaiswal/Streaming-ETL-Pipeline-using-Kafka",
    icon: faStream,
    color: "from-indigo-600 to-blue-500"
  }
];

const experience = [
  {
    role: "Graduate Student Researcher",
    date: "Apr 2024 - Present",
    company: "WiFire Lab, SDSC, UCSD",
    location: "San Diego, CA",
    desc: [
      "Architected a High-Throughput Data Pipeline for a 234GB spatiotemporal dataset, optimizing ingestion for Transformer (ViViT) models.",
      "Optimized Computational Efficiency by implementing O(N log N) Fourier-domain mixing algorithms.",
      "Engineered a 'Differentiable Simulation' Framework (Soft-ROS), integrating physical constraints directly into the Gradient Flow.",
      "Standardized Scientific Workflows using MLflow, establishing a robust Model Registry."
    ]
  },
  {
    role: "Software Engineer IC2",
    date: "July 2021 - Sep 2023",
    company: "Oracle",
    location: "Bangalore, India",
    desc: [
      "Scaled and optimized the Distributed Control Plane for a global Dynamic Routing System across 18+ regions.",
      "Engineered a 'Bandwidth Broker' POC using Aerospike and Redis to process 1M+ metric updates/sec.",
      "Developed the 'Fleet Health Visualizer,' a Go-based monitoring tool utilizing real-time telemetry.",
      "Engineered a 'Developer Control Plane' through standardized SDKs and Terraform providers."
    ]
  },
  {
    role: "MLOps Engineer",
    date: "May 2020 - Feb 2021",
    company: "Proton AutoML (Acquired by Cliently)",
    location: "Remote, US",
    desc: [
      "Led the architectural evolution of core AutoML services into a resilient Microservices Architecture.",
      "Architected a cost-optimized, self-managed Kubernetes platform for high-density ML workloads.",
      "Developed Custom Kubernetes Operators (CRDs) in Go to automate the lifecycle of training jobs.",
      "Hardened the Inference Data Plane by migrating to a high-concurrency architecture using Nginx and AWS ALBs."
    ]
  }
];

// --- MAIN COMPONENT ---

export default function Home() {
  const { theme, setTheme } = useTheme();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Form Handling Logic (Replaces your old script)
  const handleContactSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const name = formData.get('name');
    const email = formData.get('email');
    const subject = formData.get('subject');
    const message = formData.get('message');

    if (!name || !email || !subject || !message) {
      alert('Please fill out all fields.');
      return;
    }

    const recipientEmail = 'jjaiswal@ucsd.edu';
    const body = `Name: ${name}\nSender's Email: ${email}\n\nMessage:\n${message}`;
    const mailtoLink = `mailto:${recipientEmail}?subject=${encodeURIComponent(subject as string)}&body=${encodeURIComponent(body)}`;
    
    window.location.href = mailtoLink;
  };

  return (
    <main className="min-h-screen bg-gray-50 text-gray-800 dark:bg-dark-900 dark:text-gray-100 font-sans transition-colors duration-300">
      
      {/* Navigation */}
      <nav className="fixed w-full bg-white/80 dark:bg-dark-900/80 backdrop-blur-md z-50 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
                <div className="flex items-center">
                    <a href="#home" className="text-xl font-bold text-primary-600 dark:text-primary-400">Jaynil Jaiswal</a>
                </div>
                
                {/* Desktop Menu */}
                <div className="hidden md:flex items-center space-x-8">
                    {['Home', 'About', 'Projects', 'Experience', 'Blog', 'AI Search', 'Contact'].map((item) => (
                      <a key={item} href={`#${item.toLowerCase().replace(' ', '')}`} className="relative nav-link text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                        {item}
                      </a>
                    ))}
                    
                    {/* Dark Mode Toggle */}
                    <button 
                      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                      className="p-2 rounded-full bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-all"
                      aria-label="Toggle Dark Mode"
                    >
                      <FontAwesomeIcon icon={theme === 'dark' ? faSun : faMoon} />
                    </button>
                </div>

                {/* Mobile Menu Button */}
                <div className="md:hidden flex items-center">
                    <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="text-gray-700 dark:text-gray-300 p-2">
                        <FontAwesomeIcon icon={faBars} size="lg" />
                    </button>
                </div>
            </div>
        </div>
        
        {/* Mobile Menu Dropdown */}
        {mobileMenuOpen && (
           <div className="md:hidden pb-4 px-4 bg-white dark:bg-dark-900 shadow-lg">
              <div className="flex flex-col space-y-3 pt-2">
                 {['Home', 'About', 'Projects', 'Experience', 'Blog', 'AI Search', 'Contact'].map((item) => (
                      <a 
                        key={item} 
                        href={`#${item.toLowerCase().replace(' ', '')}`} 
                        className="block py-2 text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400"
                        onClick={() => setMobileMenuOpen(false)}
                      >
                        {item}
                      </a>
                  ))}
              </div>
           </div>
        )}
      </nav>

      {/* Hero Section */}
      <section id="home" className="min-h-screen flex items-center pt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
            <div className="flex flex-col md:flex-row items-center justify-between">
                <div className="md:w-1/2 mb-12 md:mb-0 animate-fade-in">
                    <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-4">
                        Hi, I'm <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-emerald-500">Jaynil Jaiswal</span>
                    </h1>
                    <h2 className="text-2xl md:text-3xl text-gray-600 dark:text-gray-300 mb-6">
                        Software Developer | ML Engineer
                    </h2>
                    <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 max-w-lg">
                        Specializing in AI Infrastructure, Orchestration, and Data Engineering.
                    </p>
                    <div className="flex space-x-4">
                        <a href="#projects" className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors duration-300 shadow-md hover:shadow-lg">
                            View My Work
                        </a>
                        <a href="#contact" className="px-6 py-3 border border-primary-600 text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-dark-800 rounded-lg transition-colors duration-300">
                            Contact Me
                        </a>
                    </div>
                </div>
                <div className="md:w-1/2 flex justify-center animate-float">
                    <div className="relative w-64 h-64 md:w-80 md:h-80 lg:w-96 lg:h-96">
                        <div className="absolute inset-0 bg-gradient-to-br from-primary-400 to-primary-600 rounded-full opacity-20 blur-xl"></div>
                        <div className="relative w-full h-full flex items-center justify-center">
                            {/* Make sure img12.jpg is in public/images/ folder */}
                            <img 
                              src="/images/img12.jpg" 
                              alt="Jaynil Jaiswal" 
                              className="w-3/4 h-3/4 object-cover rounded-full border-4 border-white dark:border-dark-800 shadow-lg"
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-20 bg-gray-100 dark:bg-dark-800">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-center mb-12">
                <span className="border-b-4 border-primary-600 pb-2">About Me</span>
            </h2>
            
            <div className="flex flex-col md:flex-row items-center gap-12">
                <div className="md:w-1/3 flex justify-center">
                    <div className="w-64 h-64 rounded-full overflow-hidden border-4 border-primary-600 shadow-lg">
                        <img 
                          src="https://images.unsplash.com/photo-1571171637578-41bc2dd41cd2?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1470&q=80" 
                          alt="Jaynil Jaiswal" 
                          className="w-full h-full object-cover"
                        />
                    </div>
                </div>
                
                <div className="md:w-2/3">
                    <p className="text-lg mb-6 leading-relaxed">
                        I'm a Software Developer and Machine Learning Engineer with a focus on building robust AI control planes, orchestration platforms, and scalable data infrastructure. 
                        My expertise spans from low-level systems programming in C++ and Go to architecting distributed microservices and RAG pipelines.
                    </p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                        <div>
                            <h3 className="text-xl font-semibold mb-3 text-primary-600 dark:text-primary-400">Education</h3>
                            <ul className="space-y-3">
                                <li className="flex items-start">
                                    <span className="mr-2">🎓</span>
                                    <span>
                                        <strong>M.S. in Computer Science</strong><br/>
                                        UC San Diego, 2023 - 2025
                                    </span>
                                </li>
                                <li className="flex items-start">
                                    <span className="mr-2">🎓</span>
                                    <span>
                                        <strong>B.Tech in Computer Science</strong><br/>
                                        Indian Institute of Technology Roorkee, 2021
                                    </span>
                                </li>
                            </ul>

                            <h3 className="text-xl font-semibold mt-6 mb-3 text-primary-600 dark:text-primary-400">Certifications</h3>
                            <ul className="space-y-2">
                                <li className="flex items-start">
                                    <span className="mr-2">📜</span>
                                    <span>
                                        <strong>IBM Data Engineering Professional</strong><br/>
                                        Issued Apr 2025
                                    </span>
                                </li>
                            </ul>
                        </div>
                        
                        <div>
                            <h3 className="text-xl font-semibold mb-3 text-primary-600 dark:text-primary-400">Skills</h3>
                            <div className="flex flex-wrap gap-2">
                                {["Java (Spring Boot)", "Python (PyTorch, TensorFlow)", "Go", "C++", "Temporal.io", "Kubernetes", "KServe", "Redis/Aerospike", "Terraform"].map(skill => (
                                    <span key={skill} className="px-3 py-1 bg-primary-100 dark:bg-primary-900/50 text-primary-800 dark:text-primary-200 rounded-full text-sm">
                                        {skill}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                    
                    <div className="flex space-x-4">
                        <a href="#contact" className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors duration-300">
                            Get In Touch
                        </a>
                        <a href="#" className="px-6 py-3 border border-primary-600 text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-dark-700 rounded-lg transition-colors duration-300 flex items-center">
                            <FontAwesomeIcon icon={faFilePdf} className="mr-2" /> Download CV
                        </a>
                    </div>
                </div>
            </div>
        </div>
      </section>

      {/* Projects Section */}
      <section id="projects" className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-center mb-12">
                <span className="border-b-4 border-primary-600 pb-2">Featured Projects</span>
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {projects.map((project, index) => (
                    <div key={index} className="bg-white dark:bg-dark-800 rounded-xl shadow-md overflow-hidden hover:-translate-y-2 hover:shadow-xl transition-all duration-300">
                        <div className={`h-48 bg-gradient-to-r ${project.color} flex items-center justify-center`}>
                            <FontAwesomeIcon icon={project.icon} className="text-white text-6xl" />
                        </div>
                        <div className="p-6">
                            <h3 className="text-xl font-bold mb-2">{project.title}</h3>
                            <p className="text-gray-600 dark:text-gray-300 mb-4 text-sm">
                                {project.desc}
                            </p>
                            <div className="flex flex-wrap gap-2 mb-4">
                                {project.tags.map(tag => (
                                    <span key={tag} className="px-2 py-1 bg-gray-100 dark:bg-dark-700 text-gray-800 dark:text-gray-200 rounded-full text-xs">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                            <a href={project.link} target="_blank" rel="noopener noreferrer" className="text-primary-600 dark:text-primary-400 hover:underline flex items-center">
                                <FontAwesomeIcon icon={project.link.includes('github') ? faGithub : faExternalLinkAlt} className="mr-2" /> 
                                {project.link === '#' ? 'View Details' : 'View Code / Demo'}
                            </a>
                        </div>
                    </div>
                ))}
            </div>
            
            <div className="text-center mt-12">
                <a href="https://github.com/JaynilJaiswal" target="_blank" rel="noopener noreferrer"
                className="px-6 py-3 border border-primary-600 text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-dark-700 rounded-lg transition-colors duration-300 flex items-center justify-center mx-auto w-fit">
                    <FontAwesomeIcon icon={faGithub} className="mr-2" /> View All Projects
                </a>
            </div>
        </div>
      </section>

      {/* Experience Section */}
      <section id="experience" className="py-20 bg-gray-100 dark:bg-dark-800">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-center mb-12">
                <span className="border-b-4 border-primary-600 pb-2">Professional Experience</span>
            </h2>
            
            <div className="space-y-8">
                {experience.map((exp, index) => (
                    <div key={index} className="bg-white dark:bg-dark-700 rounded-xl shadow-md p-6 hover:shadow-lg transition-shadow">
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                            <h3 className="text-xl font-bold">{exp.role}</h3>
                            <span className="text-primary-600 dark:text-primary-400 font-medium">{exp.date}</span>
                        </div>
                        <div className="flex items-center mb-4">
                            <div className="w-12 h-12 bg-primary-100 dark:bg-primary-900/50 rounded-full flex items-center justify-center mr-4">
                                <FontAwesomeIcon icon={faBuilding} className="text-primary-600 dark:text-primary-400 text-xl" />
                            </div>
                            <div>
                                <h4 className="font-semibold">{exp.company}</h4>
                                <p className="text-gray-600 dark:text-gray-300 text-sm">{exp.location}</p>
                            </div>
                        </div>
                        <ul className="list-disc pl-5 space-y-2 text-gray-700 dark:text-gray-300">
                            {exp.desc.map((item, i) => (
                                <li key={i}>{item}</li>
                            ))}
                        </ul>
                    </div>
                ))}
            </div>
        </div>
      </section>

      {/* Blog Section */}
      <section id="blog" className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-center mb-12">
                <span className="border-b-4 border-primary-600 pb-2">Latest Insights</span>
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                <div className="bg-white dark:bg-dark-800 rounded-xl shadow-md overflow-hidden hover:-translate-y-2 transition-transform">
                    <div className="h-48 bg-gradient-to-r from-blue-400 to-blue-600 flex items-center justify-center">
                        <FontAwesomeIcon icon={faRobot} className="text-white text-6xl" />
                    </div>
                    <div className="p-6">
                        <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-2">
                            <span>Aug 15, 2025</span>
                            <span className="mx-2">•</span>
                            <span>18 min read</span>
                        </div>
                        <h3 className="text-xl font-bold mb-3">MLOps Platforms: The 2025 CTO’s Guide</h3>
                        <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
                            Introduction — The True Cost of Technical Debt in Machine Learning...
                        </p>
                        <a href="https://medium.com/@jjaynil/f4f10e27bf64" target="_blank" className="text-primary-600 dark:text-primary-400 hover:underline flex items-center">
                            Read More <FontAwesomeIcon icon={faArrowRight} className="ml-2" />
                        </a>
                    </div>
                </div>

                <div className="bg-white dark:bg-dark-800 rounded-xl shadow-md overflow-hidden hover:-translate-y-2 transition-transform">
                    <div className="h-48 bg-gradient-to-r from-purple-400 to-purple-600 flex items-center justify-center">
                        <FontAwesomeIcon icon={faNetworkWired} className="text-white text-6xl" />
                    </div>
                    <div className="p-6">
                        <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-2">
                            <span>May 4, 2025</span>
                            <span className="mx-2">•</span>
                            <span>4 min read</span>
                        </div>
                        <h3 className="text-xl font-bold mb-3">Building an Efficient RAG System</h3>
                        <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
                            The ability to efficiently search and retrieve relevant information from massive datasets is a cornerstone of modern AI...
                        </p>
                        <a href="https://medium.com/@jjaynil/building-an-efficient-rag-system-mistral-chromadb-and-langchain-for-large-scale-document-af249559e775" target="_blank" className="text-primary-600 dark:text-primary-400 hover:underline flex items-center">
                            Read More <FontAwesomeIcon icon={faArrowRight} className="ml-2" />
                        </a>
                    </div>
                </div>

                <div className="bg-white dark:bg-dark-800 rounded-xl shadow-md overflow-hidden hover:-translate-y-2 transition-transform">
                    <div className="h-48 bg-gradient-to-r from-green-400 to-green-600 flex items-center justify-center">
                        <FontAwesomeIcon icon={faCloud} className="text-white text-6xl" />
                    </div>
                    <div className="p-6">
                        <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-2">
                            <span>Aug 9, 2025</span>
                            <span className="mx-2">•</span>
                            <span>5 min read</span>
                        </div>
                        <h3 className="text-xl font-bold mb-3">RAG Isn’t Dead, It’s Evolving</h3>
                        <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
                            If you’re building with AI, you know that Large Language Models (LLMs) are incredibly powerful. But you also know their limitations...
                        </p>
                        <a href="https://medium.com/@jjaynil/rag-isnt-dead-it-s-evolving-your-guide-to-the-new-ai-stack-7fddb714e418" target="_blank" className="text-primary-600 dark:text-primary-400 hover:underline flex items-center">
                            Read More <FontAwesomeIcon icon={faArrowRight} className="ml-2" />
                        </a>
                    </div>
                </div>
            </div>
            
            <div className="text-center mt-12">
                <a href="#" className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors duration-300 inline-flex items-center">
                    <FontAwesomeIcon icon={faBookOpen} className="mr-2" /> View All Articles
                </a>
            </div>
        </div>
      </section>

      {/* LLM Search Section */}
      <section id="llm" className="py-20 bg-gray-100 dark:bg-dark-800">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-center mb-12">
                <span className="border-b-4 border-primary-600 pb-2">AI-Powered Portfolio Search</span>
            </h2>
            
            <div className="bg-white dark:bg-dark-700 rounded-xl shadow-lg p-6">
                <div className="mb-8 text-center">
                    <p className="text-lg text-gray-700 dark:text-gray-300 max-w-3xl mx-auto">
                        Explore my AI-powered portfolio search engine — ask questions and discover my projects interactively using a Large Language Model.
                    </p>
                </div>
                
                <div className="relative h-[600px] w-full">
                    <iframe 
                        id="llm-iframe" 
                        src="https://jayniljaiswal.duckdns.org" 
                        className="w-full h-full rounded-lg border-none"
                        allow="accelerometer; ambient-light-sensor; camera; encrypted-media; geolocation; gyroscope; hid; microphone; midi; payment; usb; vr; xr-spatial-tracking" 
                        sandbox="allow-forms allow-modals allow-popups allow-presentation allow-same-origin allow-scripts"
                    />
                </div>
            </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="py-20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl font-bold text-center mb-12">
                <span className="border-b-4 border-primary-600 pb-2">Get In Touch</span>
            </h2>
            
            <div className="flex flex-col lg:flex-row gap-12">
                <div className="lg:w-1/2">
                    <h3 className="text-2xl font-semibold mb-6">Let's collaborate</h3>
                    <p className="text-gray-600 dark:text-gray-300 mb-8">
                        I'm always interested in hearing about new projects, research opportunities, or just chatting about AI and software engineering. 
                    </p>
                    
                    <div className="space-y-6">
                        <div className="flex items-start">
                            <div className="w-12 h-12 bg-primary-100 dark:bg-primary-900/50 rounded-full flex items-center justify-center mr-4">
                                <FontAwesomeIcon icon={faEnvelope} className="text-primary-600 dark:text-primary-400 text-xl" />
                            </div>
                            <div>
                                <h4 className="font-semibold">Email</h4>
                                <a href="mailto:jjaiswal@ucsd.edu" className="text-gray-600 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400">jjaiswal@ucsd.edu</a>
                            </div>
                        </div>
                        
                        <div className="flex items-start">
                            <div className="w-12 h-12 bg-primary-100 dark:bg-primary-900/50 rounded-full flex items-center justify-center mr-4">
                                <FontAwesomeIcon icon={faMapMarkerAlt} className="text-primary-600 dark:text-primary-400 text-xl" />
                            </div>
                            <div>
                                <h4 className="font-semibold">Location</h4>
                                <p className="text-gray-600 dark:text-gray-300">Raleigh, NC</p>
                            </div>
                        </div>
                        
                        <div className="flex items-center space-x-6 pt-4">
                            <a href="https://www.linkedin.com/in/jaynil-jaiswal/" target="_blank" className="w-10 h-10 bg-gray-100 dark:bg-dark-700 rounded-full flex items-center justify-center text-gray-700 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-primary-900/50 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                                <FontAwesomeIcon icon={faLinkedinIn} size="lg" />
                            </a>
                            <a href="https://github.com/JaynilJaiswal" target="_blank" className="w-10 h-10 bg-gray-100 dark:bg-dark-700 rounded-full flex items-center justify-center text-gray-700 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-primary-900/50 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                                <FontAwesomeIcon icon={faGithub} size="lg" />
                            </a>
                            <a href="#" className="w-10 h-10 bg-gray-100 dark:bg-dark-700 rounded-full flex items-center justify-center text-gray-700 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-primary-900/50 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                                <FontAwesomeIcon icon={faTwitter} size="lg" />
                            </a>
                            <a href="#" className="w-10 h-10 bg-gray-100 dark:bg-dark-700 rounded-full flex items-center justify-center text-gray-700 dark:text-gray-300 hover:bg-primary-100 dark:hover:bg-primary-900/50 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                                <FontAwesomeIcon icon={faMediumM} size="lg" />
                            </a>
                        </div>
                    </div>
                </div>
                
                <div className="lg:w-1/2">
                    <form className="space-y-6" onSubmit={handleContactSubmit}>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <label htmlFor="name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Name</label>
                                <input name="name" type="text" id="name" required className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-dark-600 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:bg-dark-700 dark:text-gray-100 transition-colors" placeholder="Your name" />
                            </div>
                            <div>
                                <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Email</label>
                                <input name="email" type="email" id="email" required className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-dark-600 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:bg-dark-700 dark:text-gray-100 transition-colors" placeholder="Your email" />
                            </div>
                        </div>
                        
                        <div>
                            <label htmlFor="subject" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Subject</label>
                            <input name="subject" type="text" id="subject" required className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-dark-600 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:bg-dark-700 dark:text-gray-100 transition-colors" placeholder="Subject" />
                        </div>
                        
                        <div>
                            <label htmlFor="message" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Message</label>
                            <textarea name="message" id="message" rows={5} required className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-dark-600 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:bg-dark-700 dark:text-gray-100 transition-colors" placeholder="Your message"></textarea>
                        </div>
                        
                        <button type="submit" className="w-full px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors duration-300">
                            Send Message
                        </button>
                    </form>
                </div>
            </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-100 dark:bg-dark-800 py-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col items-center">
                <a href="#home" className="text-2xl font-bold text-primary-600 dark:text-primary-400 mb-4">Jaynil Jaiswal</a>
                <p className="text-gray-600 dark:text-gray-300 mb-6 text-center max-w-md">
                    Building intelligent systems and scalable solutions to solve real-world problems.
                </p>
                
                <div className="flex space-x-6 mb-8">
                    <a href="https://www.linkedin.com/in/jaynil-jaiswal" className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                        <FontAwesomeIcon icon={faLinkedinIn} size="lg" />
                    </a>
                    <a href="https://github.com/JaynilJaiswal" className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                        <FontAwesomeIcon icon={faGithub} size="lg" />
                    </a>
                    <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                        <FontAwesomeIcon icon={faTwitter} size="lg" />
                    </a>
                    <a href="#" className="text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                        <FontAwesomeIcon icon={faMediumM} size="lg" />
                    </a>
                </div>
                
                <div className="border-t border-gray-200 dark:border-dark-700 w-full pt-6">
                    <p className="text-gray-500 dark:text-gray-400 text-sm text-center">
                        &copy; 2025 Jaynil Jaiswal. All rights reserved.
                    </p>
                </div>
            </div>
        </div>
      </footer>
    </main>
  );
}