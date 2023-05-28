import styles from './AllObjects.module.scss';
import store from '../../store/store';
import { useEffect, useState } from 'react';
import CustomButton from '../CustomButton/CustomButton';
import { observer } from 'mobx-react-lite';

const AllObjects = observer(({ title, openCreateModal = () => {}, handleSelectObject = () => {} }) => {
	const { allObjects, authUser } = store;
	const [visibleObj, setVisibleObj] = useState([]);
	const [search, setSearch] = useState('');

	useEffect(() => {
		setVisibleObj(allObjects);
	}, [allObjects]);

	useEffect(() => {
		setTimeout(() => {
			filterObjects();
		}, 1000);
	}, [search]);

	const filterObjects = () => {
		if(search === '') return;
		const filteredObj = visibleObj.filter(({ name }) => {
			return name.toLowerCase().includes(search.toLowerCase());
		});
		setVisibleObj(filteredObj);
	};

	return (
		<div className={styles.container}>
			<div className={styles.title}>
				{authUser === 'admin' && <CustomButton name="+ Добавить объект" handleClick={openCreateModal} />}

				{title}

				<input
					className={styles.inputItem}
					placeholder="Поиск..."
					value={search}
					onChange={(e) => setSearch(e.target.value)}
				/>
			</div>

			<div className={styles.cards}>
				{visibleObj.map(({ id, name, photo }) => {
					return (
						<div key={id} className={styles.card} onClick={() => handleSelectObject(id)}>
							<img className={styles.photo} src={photo} alt="" />

							<div className={styles.name}>{name}</div>
						</div>
					);
				})}
			</div>
		</div>
	);
});

export default AllObjects;
