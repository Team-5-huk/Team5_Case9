import styles from './AllObjects.module.scss';
import store from '../../store/store';
import { useState } from 'react';
import CustomButton from '../CustomButton/CustomButton';
import { observer } from 'mobx-react-lite';
import { Box } from '@mui/material';

const AllObjects = observer(({ title, openCreateModal = () => {}, handleSelectObject = () => {} }) => {
	const { allObjects, authUser } = store;
	const [search, setSearch] = useState('');

	const searchObject = (object) => object.name.toLowerCase().includes(search.toLowerCase());

	return (
		<div className={styles.container}>
			<div className={styles.title}>
				<div style={{marginBottom: '10px'}}>{title}</div>

				<input
					className={styles.inputItem}
					placeholder="Поиск..."
					value={search}
					onChange={(e) => setSearch(e.target.value)}
				/>
			</div>


			<Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
				{allObjects.filter(searchObject).map(({ id, name, photo }) => {
					return (
						<div key={id} className={styles.card} onClick={() => handleSelectObject(id)}>
							<img className={styles.photo} src={photo} alt="" />

							<div className={styles.name}>{name}</div>
						</div>
					);
				})}
			</Box>
			
			{authUser === 'admin' && <CustomButton name="+ Добавить объект" width={'160px'} handleClick={openCreateModal} />}
		</div>
	);
});

export default AllObjects;
