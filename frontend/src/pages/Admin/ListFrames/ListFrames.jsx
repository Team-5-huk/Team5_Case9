import React, { useState, useEffect } from 'react';
import styles from './ListFrames.module.scss';
import store from '../../../store/store';
import CreateFrame from './CreateFrame/CreateFrame';
import ViewModal from '../../../components/ViewModal/ViewModal';
import CustomButton from '../../../components/CustomButton/CustomButton';
import { useNavigate, useParams } from 'react-router-dom';
import ListCompleted from '../../../components/ListCompleted/ListCompleted';
import { observer } from 'mobx-react-lite';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import linksStore from '../../../store/linksStore';
import {apiGetProjects} from '../../../api/api';

const columns = [
	{ field: 'id', headerName: 'ID' },
	{ field: 'name', headerName: 'Имя', flex: 1, cellClassName: 'name-column--cell' },
	{ field: 'lastName', headerName: 'Фамилия', flex: 1, cellClassName: 'name-column--cell' },
	{ field: 'email', headerName: 'Email', flex: 1 },
	{ field: 'phone', headerName: 'Телефон', flex: 1 },
	{ field: 'date', headerName: 'Последний обход', flex: 1 },
];

const ListFrames = observer(() => {
	const { frames, setFrames, employees } = store;
	const {linkGetProjects} = linksStore;
	const { object } = useParams();
	const navigate = useNavigate();
	const [isOpenModal, setOpenModal] = useState(false);

	useEffect(() => {
		getBuildings(`${linkGetProjects}${object}/getbuildings/`);
	}, []);

	const getBuildings = (url = '') => {
		apiGetProjects(url).then(({data, error}) => {
			setFrames(data.map(({floors_total, name, plan, samolet_pk}) => {
				return {
					id: samolet_pk,
					name,
					floors: floors_total,
					photo: plan,
					isShow: false
				}
			}));
			// console.log(data);
			console.log(error);
		})
	};

	const openCreateModal = () => {
		setOpenModal(true);
	};

	const closeCreateModal = () => {
		setOpenModal(false);
	};

	return (
		<div className={styles.container}>
			<div className={styles.title}>
				<ArrowBackIosNewIcon sx={{ color: '#007bfb', cursor: 'pointer' }} onClick={() => navigate('/admin')} />
				Квартал Строгино
			</div>

			<div className={styles.frames}>
				{frames.map(({ id, name, floors, photo }) => {
					return (
						<div key={id} className={styles.frame} onClick={() => navigate(`/admin/${object}/${id}`)}>
							{name}
							<div>Кол-во этажей: {floors}</div>
							<img style={{width: '100px'}} src={photo}/>
						</div>
					);
				})}
			</div>

			<CustomButton name="+ Добавить корпус" handleClick={openCreateModal} />

			<ViewModal title="Новый корпус" isModal={isOpenModal} closeModal={closeCreateModal}>
				<CreateFrame closeModal={closeCreateModal} />
			</ViewModal>

			<div className={styles.blockInfo}>
				Обходчики на объекте
				<ListCompleted columns={columns} data={employees} />
			</div>
		</div>
	);
});

export default ListFrames;
