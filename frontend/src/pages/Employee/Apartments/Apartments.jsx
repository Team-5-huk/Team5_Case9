import React, { useState, useEffect } from 'react';
import styles from './Apartments.module.scss';
import { useParams, useNavigate } from 'react-router-dom';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import { observer } from 'mobx-react-lite';
import linksStore from '../../../store/linksStore';
import { apiGetProjects } from '../../../api/api';
import store from '../../../store/store';

const Apartments = observer(() => {
	const { linkGetFlats } = linksStore;
	const { apartments, setApartments } = store;
	const navigate = useNavigate();
	const { object, frame, section } = useParams();
	const [floors, setFloors] = useState([]);

	useEffect(() => {
		getFlats(`${linkGetFlats}${section}/getflats/`);
	}, []);

	const getFlats = (url = '') => {
		apiGetProjects(url).then(({ data, error }) => {
			setApartments(
				data
					.reduce((acc, { floor, number }) => {
						const floorObj = acc.find((f) => f.floor === floor);
						if (!floorObj) {
							acc.push({ id: acc.length + 1, floor, apartaments: [`${number}`] });
						} else {
							floorObj.apartaments.push(`${number}`);
						}
						return acc;
					}, [])
					.map(({ id, floor, apartaments }) => ({ id, floor, apartaments }))
					.reverse()
			);
			console.log(error);
		});
	};

	return (
		<div className={styles.container}>
			<div className={styles.title}>
				<ArrowBackIosNewIcon
					sx={{ color: '#007bfb', cursor: 'pointer' }}
					onClick={() => navigate(`/employee/${object}/${frame}`)}
				/>
				Выберите квартиру для обхода
			</div>

			<div className={styles.list}>
				{apartments.map(({ floor, apartaments }) => {
					return (
						<div className={styles.floor} key={floor}>
							{floor}
							{apartaments.map((apartament) => {
								return (
									<div
										className={styles.apartament}
										key={apartament}
										onClick={() => navigate(`/employee/${object}/${frame}/${section}/${apartament}/camera`)}
									>
										{apartament}
									</div>
								);
							})}
						</div>
					);
				})}
			</div>
		</div>
	);
});

export default Apartments;
