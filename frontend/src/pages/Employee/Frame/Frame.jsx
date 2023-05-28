import React, {useEffect} from 'react';
import styles from './Frame.module.scss';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { ListItemButton, ListItemText, Collapse, List } from '@mui/material';
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

	return (
		<div className={styles.container}>
			<div className={styles.title}>
				<ArrowBackIosNewIcon sx={{ color: '#007bfb', cursor: 'pointer' }} onClick={() => navigate('/employee')} />
				Выберите корпус
			</div>

			<div className={styles.frames}>
				{frames.map(({ id, name, floors, photo }) => {
					return (
						<div key={id} className={styles.frame} onClick={() => navigate(`/employee/${object}/${id}`)}>
							{name}
							<div>Кол-во этажей: {floors}</div>
							<img style={{width: '100px'}} src={photo}/>
						</div>
					);
				})}
			</div>

			{/* {frames.map(({ id, name, sections, isShow }) => {
				return (
					<List key={id}>
						<ListItemButton key={id} onClick={() => setVisibleListFrame(id)}>
							<ListItemText primary={name} />
							{isShow ? <ExpandLess /> : <ExpandMore />}
						</ListItemButton>

						<Collapse in={isShow} timeout="auto" unmountOnExit>
							<List component="div" disablePadding>
								{sections.map((section) => {
									return (
										<ListItemButton
											key={section}
											sx={{ pl: 4 }}
											onClick={() => navigate(`/employee/${object}/${id}`)}
										>
											<ListItemText primary={section} />
										</ListItemButton>
									);
								})}
							</List>
						</Collapse>
					</List>
				);
			})} */}
		</div>
	);
});

export default Frame;
