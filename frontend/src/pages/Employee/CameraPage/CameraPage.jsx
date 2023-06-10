import React, { useState, useRef, useEffect } from 'react';
import styles from './CameraPage.module.scss';
import ViewModal from '../../../components/ViewModal/ViewModal';
import Instruction from '../Instruction/Instruction';
import CustomButton from '../../../components/CustomButton/CustomButton';
import { useParams, useNavigate } from 'react-router-dom';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import { observer } from 'mobx-react-lite';
import { apiPostFile } from '../../../api/employeeApi/employeeApi';
import linksStore from '../../../store/linksStore';
import store from '../../../store/store';
import { apiGetProjects } from '../../../api/api';
import ListCompleted from '../../../components/ListCompleted/ListCompleted';
import { Box } from '@mui/material';
import PieChart from '../../../components/PieChart/PieChart';

const columns = [
	{ field: 'id', headerName: 'Номер обхода', flex: 1 },
	{ field: 'date', headerName: 'Дата обхода', flex: 1 },
	{ field: 'is_analysed', headerName: 'Готовность анализа', flex: 1 },
];

const CameraPage = observer(() => {
	const { linkCreateVideo, linkGetRounds } = linksStore;
	const { numberFlat } = store;
	const { object, frame, section, apartament } = useParams();
	const navigate = useNavigate();
	const intervalIdRef = useRef(null);

	const [isOpenModal, setOpenModal] = useState(true);
	const [videoFile, setVideoFile] = useState({});
	const [isLoad, setLoad] = useState(false);
	const [isAnalize, setAnalize] = useState(false);
	const [video, setVideo] = useState('');
	const [rounds, setRounds] = useState([]);
	const [selectImage, setSelectImage] = useState('');
	const [dataPie, setDataPie] = useState([]);
	const [infoRound, setInfoRound] = useState({});
	const [squire, setSquire] = useState('');

	const pathLast = `/employee/${object}/${frame}/${section}`;

	useEffect(() => {
		return () => {
		  clearInterval(intervalIdRef.current);
		};
	  }, []);

	const fileUploadHandler = (event) => {
		event.preventDefault();
		event.stopPropagation();
		//Получаем все файлы из инпута
		const newFiles = event.target.files;
		console.log(newFiles);
		setVideoFile(newFiles[0]);
	};

	const closeModalInstructions = () => {
		setOpenModal(false);
	};

	//Вызов аналитика до момента, пока она не сработает
	const sendVideo = () => {
		setLoad(true);
		apiPostFile(linkCreateVideo, videoFile, parseInt(apartament)).then(({ data, error }) => {
			setLoad(false);
			setVideo(data.video);
			setAnalize(data.is_analysed);
			getAnalizFetching();

			console.log(data);
			console.log(error);
		});
	};

	const getAnalizFetching = () => {
		getRounds(`${linkGetRounds}${apartament}/getchecks/`);

		if (isAnalize) {
			console.log('Анализ завершен');
		} else {
			intervalIdRef.current = setTimeout(getAnalizFetching, 3000);
		}
	};

	const getRounds = (url = '') => {
		apiGetProjects(url).then(({ data, error }) => {
			setRounds(data);
			console.log(data[data.length - 1]);
			setAnalize(data[data.length - 1].is_analysed);
			console.log(error);

			setInfoRound(data[data.length - 1].analysis);
			
			if (!Object.keys(data[data.length - 1].analysis).length) return;

			setSquire(data[data.length - 1].analys_square);

			setSelectImage(data[data.length - 1]?.analys_image_url);

			setDataPie(
				data[data.length - 1].analysis.Detected_objects.map((item) => {
					const { Frame_count, Name, Score } = item;

					return {
						id: `${Score} (кадров: ${Frame_count})`,
						label: Name,
						value: Score,
						color: `hsl(2${Score}, 70%, 50%)`,
					};
				})
			);
		});
	};

	return (
		<div className={styles.container}>
			<div className={styles.title}>
				<ArrowBackIosNewIcon sx={{ color: '#007bfb', cursor: 'pointer' }} onClick={() => navigate(pathLast)} />
				Квартира № {numberFlat}
			</div>

			<div>
				<div style={{ fontSize: '18px', color: '#007bfb' }}>Загрузка данных о квартире</div>
				<div className={styles.disk}>
					<div className={styles.disk_btns}>
						<div className="disk_upload">
							<label htmlFor="disk_upload-input" className={styles.diskUpload}>
								Загрузить файл
							</label>
							<input
								multiple={true}
								onChange={(event) => fileUploadHandler(event)}
								type="file"
								id="disk_upload-input"
								className={styles.diskInput}
							/>
						</div>
					</div>
				</div>
				<CustomButton name={'Отправить видео'} width={'150px'} handleClick={sendVideo} />
				{isLoad && <div>Загрузка файлов на сервер, подождите...</div>}
				{isAnalize ? <div>Анализ завершен</div> : video && <div>Анализ видео...</div>}
			</div>





			{!!Object.keys(infoRound).length && (
				<div className={styles.infoBlock}>
					<div style={{ fontSize: '20px' }}>Подробная информация о квартире</div>

					<div style={{ fontSize: '16px' }}>Проанализированная площадь: {squire} м2</div>

					<div>
						Общая готовность квартиры: {infoRound.Ready_precentage}%
						<div>
							<div>
								Готовность потолка:
								{infoRound.Ceiling_ready ? <span>&#x2714;</span> : <span>&#x2716;</span>}
							</div>
							<div>
								Готовность дверей:
								{infoRound.Door_ready ? <span>&#x2714;</span> : <span>&#x2716;</span>}
							</div>

							<Box height="300px" >
								<PieChart data={dataPie} />
							</Box>
						</div>
					</div>

					<img style={{ margin: '10px 0', borderRadius: '10px' }} src={selectImage} />
				</div>
			)}

			{video && <video className={styles.backVideo} src={video} controls={true} />}

			<ListCompleted columns={columns} data={rounds} />

			<ViewModal title="Следуйте инструкции" isModal={isOpenModal} closeModal={closeModalInstructions}>
				<Instruction closeModal={closeModalInstructions} />
			</ViewModal>
		</div>
	);
});

export default CameraPage;
