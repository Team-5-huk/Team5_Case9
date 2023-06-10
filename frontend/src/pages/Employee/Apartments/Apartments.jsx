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
	const { apartments, setApartments, setNumberFlat } = store;
	const navigate = useNavigate();
	const { object, frame, section } = useParams();

	useEffect(() => {
		getFlats(`${linkGetFlats}${section}/getflats/`);
	}, []);

	const getFlats = (url = '') => {
		apiGetProjects(url).then(({ data, error }) => {
			setApartments(
				data
					.reduce((acc, { id, floor, number, Ready_precentage, checks }) => {
						const floorObj = acc.find((f) => f.floor === floor);
						if (!floorObj) {
							acc.push({
								id: acc.length + 1,
								floor,
								apartaments: [{ id, number, Ready_precentage, checks, isSelected: false }],
							});
						} else {
							floorObj.apartaments.push({ id, number, Ready_precentage, checks, isSelected: false });
						}
						return acc;
					}, [])
					.map(({ id, floor, apartaments }) => ({ id, floor, apartaments }))
					.reverse()
			);
			console.log(error);
		});
	};

	const selectFlat = (id, number) => {
		setNumberFlat(number);
		navigate(`/employee/${object}/${frame}/${section}/${id}/camera`);
	};

	const onMouseEnter = (select) => {
		const flat = apartments.reduce((result, { apartaments }) => {
			if (!result) {
			  const found = apartaments.find((a) => a.id === select);
			  if (found) {
				return found;
			  }
			}
			return result;
		  }, null);
		
		  flat.isSelected = true;
	};

	const onMouseLeave = (select) => {
		const flat = apartments.reduce((result, { apartaments }) => {
			if (!result) {
			  const found = apartaments.find((a) => a.id === select);
			  if (found) {
				return found;
			  }
			}
			return result;
		  }, null);
		
		  flat.isSelected = false;
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
							<div style={{width: '40px', height: '40px'}}>{floor}</div>
							{apartaments.map(({ id, number, Ready_precentage, isSelected }) => {
								const t = Math.min(Math.max(Ready_precentage, 0), 100);
								const r = Math.round(255 * Math.min((100 - t) / 50, 1));
								const g = Math.round(255 * Math.min(t / 50, 1));
								const background = `rgba(${r}, ${g}, 0, 0.3)`;

								const content = isSelected ? `${Ready_precentage}%` : number;

								return (
									<div
										key={id}
										style={{ background }}
										className={styles.apartament}
										onMouseEnter={() => onMouseEnter(id)}
										onMouseLeave={() => onMouseLeave(id)}
										onClick={() => selectFlat(id, number)}
									>
										{content}
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
