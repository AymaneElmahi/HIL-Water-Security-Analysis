<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a name="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Author](https://img.shields.io/badge/author-@AymaneElmahi-blue)](https://github.com/AymaneElmahi)
[![Author](https://img.shields.io/badge/author-@Simohamed0-blue)](https://github.com/Simohamed0)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/paper-here-svg)](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09526562.pdf)

<h3 align="center">A hardware-in-the-loop Water
                    Distribution Testbed dataset for
                    cyber-physical security testing </h3>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <!-- <li><a href="#roadmap">Roadmap</a></li> -->
    <!-- <li><a href="#contributing">Contributing</a></li> -->
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <!-- <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project is a visualisation of the data collected from the Hardware-in-the-loop Water Distribution Testbed dataset for cyber-physical security testing. We will run some analysis on the data, and try to predict the anomaly in the system using machine learning.
We will also make a streamlit app to visualize the data and the results of our analysis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

<!-- python, jupyter, sklearn, pandas, streamlit -->

<!-- [![Python](https://img.shields.io/badge/python-3.9.0-blue)](https://www.python.org/downloads/release/python-390/)
[![Jupyter](https://img.shields.io/badge/jupyter-6.1.4-orange)](https://jupyter.org/)
[![Tensorflow](https://img.shields.io/badge/tensorflow-2.4.0-red)](https://www.tensorflow.org/)
[![Pandas](https://img.shields.io/badge/pandas-1.2.0-yellow)](https://pandas.pydata.org/) -->

<img align="left" alt="Python" width="50px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-plain.svg" />
<img align="left" alt="Jupyter" width="50px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original-wordmark.svg"/>        
<img align="left" alt="Pandas" width="50px" style="padding-right:10px;" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original-wordmark.svg" />


<br/><br/>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

To install all the required libraries for this project, run the following command:

```sh
pip install -r requirements.txt
```
This projet is working on `Windows`.

You have to then download the dataset that you can find in this website: [HIL-Water-Security-Analysis](https://ieee-dataport.org/open-access/hardware-loop-water-distribution-testbed-wdt-dataset-cyber-physical-security-testing)

And then you have to put the dataset `network_downsampled.csv` in `dataset/Network datatset/csv/`. You can do that by running the following command:

```sh
mv network_downsampled.csv dataset/Network\ datatset/csv/
```

You should then have a dataset named `dataset` in the main directory. It should contain the following files:

```sh
dataset
├───Network datatset
│   ├───csv
│   │      network_downsampled.csv
│   │      ...
│   │--─pcap
│   │      ...
│---Physical dataset
│   ...
│---README.xlsx
``` 

### Installation

```sh
git clone https://github.com/AymaneElmahi/HIL-Water-Security-Analysis.git
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage



To run the project, you can run streamlit app by moving to the main directory and running the following command:

```sh
streamlit run .\app\0_👋_The_Main_Page.py
```

You can also go check the work done in the jupyter notebooks, and run them if you want to.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->

## The code is divided into multiple parts:

1. `app` : contains the streamlit app
2. `dataset` : contains the data used in the project, but it is not uploaded to github because of its size
3. `notebooks` : contains the jupyter notebooks used in the project
4. `plots` : contains the plots generated by the notebooks and the streamlit app

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Aymane EL MAHI : [Message me on LinkedIn](https://www.linkedin.com/in/aymane-elmahi)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

<!-- ## Acknowledgments

- []()
- []()
- []() -->
<!--
<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/AymaneElmahi/sd-hoc.svg?style=for-the-badge
[contributors-url]: https://github.com/AymaneElmahi/sd-hoc/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AymaneElmahi/sd-hoc.svg?style=for-the-badge
[forks-url]: https://github.com/AymaneElmahi/sd-hoc/network/members
[stars-shield]: https://img.shields.io/github/stars/AymaneElmahi/sd-hoc.svg?style=for-the-badge
[stars-url]: https://github.com/AymaneElmahi/sd-hoc/stargazers
[issues-shield]: https://img.shields.io/github/issues/AymaneElmahi/sd-hoc.svg?style=for-the-badge
[issues-url]: https://github.com/AymaneElmahi/sd-hoc/issues
[license-shield]: https://img.shields.io/github/license/AymaneElmahi/sd-hoc.svg?style=for-the-badge
[license-url]: https://github.com/AymaneElmahi/sd-hoc/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/aymane-elmahi
[product-screenshot]: images/about_the_project_screenshot.png
[Flowchart]: images/flowchart.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
