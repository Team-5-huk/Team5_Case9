import React, { useEffect } from 'react';
import styles from './Frame.module.scss';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { ListItemButton, ListItemText, Collapse, List, Box } from '@mui/material';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';
import { observer } from 'mobx-react-lite';
import store from '../../../store/store';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import { apiGetProjects } from '../../../api/api';
import linksStore from '../../../store/linksStore';

const Frame = observer(() => {
	const { linkGetProjects } = linksStore;
	const { frames, setFrames, setVisibleListFrame } = store;
	const { object } = useParams();
	const navigate = useNavigate();

	useEffect(() => {
		getBuildings(`${linkGetProjects}${object}/getbuildings/`);
	}, []);

	const getBuildings = (url = '') => {
		apiGetProjects(url).then(({ data, error }) => {
			setFrames(
				data.map(({ floors_total, name, plan, samolet_pk }) => {
					return {
						id: samolet_pk,
						name,
						floors: floors_total,
						photo: plan,
						isShow: false,
					};
				})
			);
			// console.log(data);
			console.log(error);
		});
	};

	return (
		<div className={styles.container}>
			<div className={styles.title}>
				<ArrowBackIosNewIcon sx={{ color: '#007bfb', cursor: 'pointer' }} onClick={() => navigate('/employee')} />
				Выберите корпус
			</div>

			<Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
				{frames.map(({ id, name, floors, photo }) => {
					return (
						<div key={id} className={styles.frame} onClick={() => navigate(`/employee/${object}/${id}`)}>
							<img className={styles.photo} src={photo} />

							<div className={styles.titleFrame}>
								{name}
								{/* <div>Кол-во этажей: {floors}</div> */}
							</div>
						</div>
					);
				})}
			</Box>
		</div>
	);
});

export default Frame;
