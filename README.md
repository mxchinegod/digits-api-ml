<p align="center">
<a target="_blank" rel="noreferrer" href="https://www.buymeacoffee.com/alloydylan
"><img style="max-width:175px;" src="./digits2.gif">
</a>
<br>
digits-api-ml is a large suite of API endpoints that directly respond with the inferrence of a given machine learning function. It does not queue things up.<br>
</p>
<hr>

## 📝 Code Properties ✨ ![start with why](https://img.shields.io/badge/start%20with-why%3F-brightgreen.svg?style=flat) ![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat) ![HitCount](https://hits.dwyl.com/dylanalloy/digits-ui.svg?style=flat-square) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

##### The below stack describes a static request->response oriented ML API

| 📁 Library | ⚙ Purpose | 📎 Version |
| :-- | :-: | --: |
| [Python](https://python.org) | Base | 3.10.8 |
| [transformers](https://pypi.org/project/transformers/) | Machine Learning | 4.23.1 |
| [AutoDD](https://pypi.org/project/AutoDD/) | Social Media Analysis | 2.1.4 |

##### You will want to know about each of these in depth by the above order.

<br>

## 🎬 Environment ✨


##### There is presently no dynamic configuration, this is due to the way the script initalizes and thusfar there is no need for private or consumer keys.

`pip install -r requirements.txt ` <br> <br>

## 📜 Provided Scripts ✨

##### Digits Pro provides some useful scripts to help you quick start.

##### Requirements provided in `requirements.txt`.

### 💡 start

```bash
uvicorn main:app --reload --port 7000 --host=127.0.0.1
```

<br>

## 🏰 Service Mesh API ✨

##### digits-api-ml is responsible for all machine learning pre- and post-processing. 

 - 🔌 Interfacing [ http ]
     - All API routing is done through http requests. It's possible sockets play a role one day.
 - 🩺 Monitoring [ http ]
     - This API intrinsically reports to digits-api-main
 - 🧮 Preprocessing [ python ]
     - Tremendous processing occurs through this API.

<br>

## 💎 Goals ✨

##### immediate

- [x] Initialize beautiful README.md
- [x] Describe service API role
- [ ] Add husky pre-commit
- [ ] Create mesh diagram
- [x] Explain preprocessor philosophy
- [ ] Contextualize hardware requirements

##### long-term

- [ ] Docker images
- [ ] Kubernetes deployment

<br>
<p align="center">
<a target="_blank" rel="noreferrer" href="https://www.buymeacoffee.com/alloydylan
"><img style="max-width:175px;" src="https://www.buymeacoffee.com/assets/img/guidelines/download-assets-sm-3.svg"></a>
</a>
</p>
