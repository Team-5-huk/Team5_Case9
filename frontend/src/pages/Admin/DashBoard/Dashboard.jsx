import React, { useState, useEffect } from 'react';
import styles from './Dashboard.module.scss';
import { Box } from '@mui/material';
import PieChart from '../../../components/PieChart/PieChart';
import ListCompleted from '../../../components/ListCompleted/ListCompleted';
import { useNavigate, useParams } from 'react-router-dom';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ViewModal from '../../../components/ViewModal/ViewModal';
import linksStore from '../../../store/linksStore';
import { apiGetProjects } from '../../../api/api';
import { observer } from 'mobx-react-lite';
import store from '../../../store/store';

const dataResult = [
	{
		id: 1,
		label: 'Готовность',
		value: 239,
		color: 'hsl(291, 70%, 50%)',
	},
	{
		id: 2,
		label: 'Остаток работ',
		value: 170,
		color: 'hsl(162, 70%, 50%)',
	},
	{
		id: 3,
		label: 'Мусор',
		value: 322,
		color: 'hsl(104, 70%, 50%)',
	},
];

const columns = [
	{ field: 'frame', headerName: 'Подъезд', flex: 1 },
	{ field: 'floor', headerName: 'Этаж', flex: 1 },
	{ field: 'number', headerName: 'номер квартиры', flex: 1 },
	{ field: 'status', headerName: 'Статус', flex: 1 },
];

const DashBoard = observer(() => {
	const { linkGetFlats } = linksStore;
	const { apartments, setApartments } = store;
	const { object, frame, section } = useParams();
	const navigate = useNavigate();

	const [isModal, setIsModal] = useState(false);
	const [selectItem, setSelectItem] = useState({});

	useEffect(() => {
		getFlats(`${linkGetFlats}${section}/getflats/`);
	}, []);

	const getFlats = (url = '') => {
		apiGetProjects(url).then(({ data, error }) => {
			setApartments(data.map(({id, floor, number, section}) => {
				return {
					id,
					frame: section,
					floor,
					number,
					status: 'Готово',
				}
			}));
			console.log(data);
			console.log(error);
		});
	};

	const handleTableItem = ({ row: { frame, floor, number, status } }) => {
		setIsModal(true);
		setSelectItem({ frame, floor, number, status });
	};

	return (
		<Box>
			<ArrowBackIosNewIcon
				sx={{ color: '#007bfb', cursor: 'pointer' }}
				onClick={() => navigate(`/admin/${object}/${frame}`)}
			/>

			<div style={{ display: 'flex' }}>
				<Box height="50vh" width="50vw">
					<PieChart data={dataResult} />
				</Box>

				<div>Общая информация о секции</div>
			</div>

			<div className={styles.blockInfo}>
				Список квартир
				<ListCompleted columns={columns} data={apartments} handleTableItem={handleTableItem} />
			</div>

			<ViewModal title={'Готовность квартиры'} isModal={isModal} closeModal={() => setIsModal(false)}>
				<div style={{ display: 'flex', flexDirection: 'column' }}>
					<div>Квартира номер {selectItem.number}</div>
					<div>Подъезд {selectItem.frame}</div>
					<div>Этаж {selectItem.floor}</div>
					<div>Готовность квартиры: {selectItem.status}</div>
				</div>
			</ViewModal>
		</Box>
	);
});

export default DashBoard;
