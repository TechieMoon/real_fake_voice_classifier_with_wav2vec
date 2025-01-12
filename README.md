# real_fake_voice_classifier_with_wav2vec

학교에서 기말고사 프로젝트로 만든 코드입니다.

프로젝트의 주제는 가짜음성과 진짜음성을 구분하는 딥러닝 모델을 만드는 것입니다.

페이스북에서 만든 wav2vec 2.0 base모델을 파인튜닝하여 만들었습니다.

데이터는 제가 만든 것이 아니라서 깃허브에 공유할 수는 없고 코드만 공유하겠습니다.

진짜음성은 real폴더에 가짜음성은 fake폴더에 분류를 해야 이 코드를 실행할 수 있습니다.

그리고 음성파일들은 16,000Hz(16Kz)이고 wav확장자명을 가져야 wav2vec 모델에 학습시킬 수 있습니다.

코드 실행은 구글의 colab에서 실행했기 때문에 몇몇 코드들은 코랩에서 실행하지 않으면 에러가 납니다.

[data_preparation](https://github.com/TechieMoon/real_fake_voice_classifier_with_wav2vec/edit/main/data_preparation.ipynb)으로 먼저 데이터를 가지고 파이토치에서 실행할 수 있는 데이터셋으로 변경합니다.

이렇게 미리 데이터셋으로 변경하면 모델을 학습시킬 때마다 매번 데이터셋으로 바꾸지 않아도 됩니다. (데이터가 크면 이 시간이 정말로 오래 걸리기 때문에 이렇게 하는 것이 좋습니다.)

그리고 [voice classifer](https://github.com/TechieMoon/real_fake_voice_classifier_with_wav2vec/edit/main/True_and_Fake_voice_classifier.ipynb) 코드를 실행하며 모델을 돌립니다.

그러면 마지막 쉘에서 계산된 검증 데이터 정확도를 확인합니다.
