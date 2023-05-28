import React from 'react';
import styles from './InfoJC.module.scss';
import { observer } from 'mobx-react-lite';
import store from '../../../store/store';
import Box from '@mui/material/Box';
import PieChart from '../../../components/PieChart/PieChart';
// import { useParams } from 'react-router-dom';

const InfoJC = observer(() => {
	// const {id} = useParams();
	const { frames, setVisibleListFrame } = store;
	console.log(frames);
	const handlerSelect = (id) => {
		setVisibleListFrame(id);
	};

	return (
		<div>
			<div className={styles.title}>Квартал Строгино</div>

			<Box height="40vh" width="50vw">
				<PieChart />
			</Box>

			<div className={styles.list}>
				<div className={styles.names}>
					{frames.map(({ id, name, isShow }) => (
						<div
							key={id}
							className={`${styles.select} ${isShow && styles.show}`}
							onClick={() => handlerSelect(id)}
						>
							{name}
						</div>
					))}
				</div>

				<div className={styles.container}>
					{frames.map(({ id, info, isShow }) => isShow && (<div key={id}>{info}</div>))}
				</div>
			</div>
		</div>
	);
});

export default InfoJC;
