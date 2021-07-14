[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/pqtrng/f2b">
    <img src="report/images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Face to BMI</h3>

  <p align="center">
    Predict Body Mass Index from facial image using deep neural network.
    <br />
    <a href="https://github.com/pqtrng/f2b"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/pqtrng/f2b">View Demo</a>
    ·
    <a href="https://github.com/pqtrng/f2b/issues">Report Bug</a>
    ·
    <a href="https://github.com/pqtrng/f2b/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

A project of predicting BMI from facial image using deep learning:

- ResNet-50 for feature extraction
- Deep neural network for regression extracted features
- VIP_Attribute dataset is used

### Built With

- [Tensorflow](https://www.tensorflow.org/)
- [Dlib](http://dlib.net/)
- [OpenCV](https://opencv.org/)
- [Keras](https://keras.io/)
- [Albumentations](https://albumentations.ai/)
- [Python](https://www.python.org/)
- [Conda](https://docs.conda.io/en/latest/)
- [scikit-learn](https://scikit-learn.org/stable/)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Conda should be installed.

- conda

  ```sh
  bash Anaconda-latest-Linux-x86_64.sh
  ```

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/pqtrng/f2b.git
   ```

2. Create conda environment and activate

   ```sh
   conda create -n f2b python=3.8; conda activate f2b
   ```

3. Install packages

   ```sh
   pip install -r requirements.txt
   ```

### Run script

- Train model

   ```sh
   python src/train_model.py <training_type> <data_set> <output_network_type> False
   ```

- Evalute model

   ```sh
   python evaluate_model.py <training_type> <data_set> <output_network_type>
   ```

### Demo

- Local

  ```sh
  python demo.py
  ```

- Online

  ```sh
  python demo.py
  ```

### Docker

- Build

  ```sh
  docker build . -t f2b --rm
  ```

- Run container

  ```sh
  docker run --name "f2b" -it f2b
  ```

- Run train and evaluate script as normal

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/pqtrng/f2b/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Pham, Trung - [@pqtrng](https://twitter.com/pqtrng)

Project Link: [https://github.com/pqtrng/f2b](https://github.com/pqtrng/f2b)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

- [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]:  https://img.shields.io/github/contributors/pqtrng/f2b.svg?style=for-the-badge
[contributors-url]: https://github.com/pqtrng/f2b/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/pqtrng/f2b.svg?style=for-the-badge
[forks-url]: https://github.com/pqtrng/f2b/network/members
[stars-shield]: https://img.shields.io/github/stars/pqtrng/f2b.svg?style=for-the-badge
[stars-url]: https://github.com/pqtrng/f2b/stargazers
[issues-shield]: https://img.shields.io/github/issues/pqtrng/f2b.svg?style=for-the-badge
[issues-url]: https://github.com/pqtrng/f2b/issues
[license-shield]: https://img.shields.io/github/license/pqtrng/f2b.svg?style=for-the-badge
[license-url]: https://github.com/pqtrng/f2b/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/pqtrng
[product-screenshot]: report/images/bmi_process.png
